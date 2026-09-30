"""Train, evaluate and export the price model.

Split by time, not at random: the model trains on older sales, is calibrated on the next
slice, and is tested on the most recent 20% - the honest version of "predict a sale it has
not seen". Everything is compared against a deterministic baseline (the smoothed suburb +
property-type median), because a model that cannot beat a median lookup is not worth shipping.

Usage:  python -m src.model.train
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.load import load
from src.features.build import FEATURES, fit, transform

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
REPORTS = ROOT / "reports"
SEED = 42
INTERVAL = (0.10, 0.90)  # an 80% range


def split_by_time(df: pd.DataFrame):
    n = len(df)
    a, b = int(n * 0.70), int(n * 0.80)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ape = np.abs(y_pred - y_true) / y_true
    log_err = np.log(y_pred) - np.log(y_true)
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "median_ape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "rmse_log": float(np.sqrt(np.mean(log_err**2))),
    }


def main() -> None:
    df = load()
    train, calib, test = split_by_time(df)
    spec = fit(train)
    X_tr, X_ca, X_te = (transform(d, spec) for d in (train, calib, test))
    y_tr, y_ca, y_te = (np.log(d["Price"].to_numpy()) for d in (train, calib, test))

    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        early_stopping_rounds=40,
        random_state=SEED,
        base_score=float(np.mean(y_tr)),
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ca, y_ca)], verbose=False)

    # Price range from real calibration errors (split conformal on the log scale).
    resid = y_ca - model.predict(X_ca)
    lo_q, hi_q = (float(np.quantile(resid, q)) for q in INTERVAL)

    pred_te = np.exp(model.predict(X_te))
    base_te = np.exp(X_te["suburb_type_median_log"].to_numpy())
    true_te = np.exp(y_te)
    lo_te, hi_te = pred_te * np.exp(lo_q), pred_te * np.exp(hi_q)

    results = {
        "rows": {"train": len(train), "calibration": len(calib), "test": len(test)},
        "test_period": [str(test["Date"].min().date()), str(test["Date"].max().date())],
        "train_period": [str(train["Date"].min().date()), str(train["Date"].max().date())],
        "trees": int(model.best_iteration + 1),
        "baseline_suburb_type_median": metrics(true_te, base_te),
        "model_xgboost": metrics(true_te, pred_te),
        "interval": {
            "nominal": INTERVAL[1] - INTERVAL[0],
            "coverage_on_test": float(np.mean((true_te >= lo_te) & (true_te <= hi_te))),
            "log_residual_quantiles": [lo_q, hi_q],
        },
        "by_type": {
            t: metrics(true_te[m], pred_te[m])
            for t in ["h", "t", "u"]
            if (m := (test["Type"] == t).to_numpy()).sum() > 0
        },
    }

    importance = model.get_booster().get_score(importance_type="gain")
    total = sum(importance.values()) or 1
    results["feature_gain_share"] = {
        FEATURES[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: round(v / total, 4)
        for k, v in sorted(importance.items(), key=lambda kv: -kv[1])
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    # Export: the trees as compact JSON for the browser (evaluated by web/model.js), plus
    # everything the browser needs to build features exactly as Python does.
    WEB.mkdir(exist_ok=True)
    booster = model.get_booster()
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")][: model.best_iteration + 1]

    def compact(node: dict) -> list:
        # leaf -> [value]; split -> [feature_index, threshold, yes, no, missing]
        if "leaf" in node:
            return [round(node["leaf"], 7)]
        kids = {c["nodeid"]: compact(c) for c in node["children"]}
        return [FEATURES.index(node["split"]), node["split_condition"], kids[node["yes"]], kids[node["no"]],
                "y" if node["missing"] == node["yes"] else "n"]

    raw_base = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    base = float(str(raw_base).strip("[]"))  # XGBoost 3 stores this as "[1.37E1]"
    (WEB / "trees.json").write_text(json.dumps({"base": base, "trees": [compact(t) for t in trees]},
                                               separators=(",", ":")))
    counts = train.groupby("Suburb").size().to_dict()
    suburbs = sorted(s for s in spec.suburb_geo if counts.get(s, 0) >= 5)
    (WEB / "spec.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "spec": spec.to_json(),
                "suburbs": suburbs,
                "interval_log": [lo_q, hi_q],
                "test_period": results["test_period"],
                "metrics": {
                    "model": results["model_xgboost"],
                    "baseline": results["baseline_suburb_type_median"],
                    "coverage": results["interval"]["coverage_on_test"],
                },
            }
        )
    )

    # Parity fixtures: the JS model must reproduce these Python predictions.
    sample = X_te.sample(50, random_state=SEED)
    (ROOT / "tests" / "parity_fixture.json").write_text(
        json.dumps({"X": sample.to_numpy().tolist(), "y_log": model.predict(sample).tolist()})
    )

    b, m = results["baseline_suburb_type_median"], results["model_xgboost"]
    print(f"test sales {len(test)}  ({results['test_period'][0]} to {results['test_period'][1]})")
    print(f"baseline  median error {b['median_ape']:.1%}  within 10% {b['within_10pct']:.1%}  MAE ${b['mae']:,.0f}")
    print(f"model     median error {m['median_ape']:.1%}  within 10% {m['within_10pct']:.1%}  MAE ${m['mae']:,.0f}")
    print(f"80% range coverage on test: {results['interval']['coverage_on_test']:.1%}   trees: {results['trees']}")


if __name__ == "__main__":
    main()
