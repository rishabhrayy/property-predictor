"""Checks that guard the pipeline's promises: no leakage, no NaNs, beats the baseline,
an honest price range, and a browser model that matches Python exactly."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES, fit, transform

ROOT = Path(__file__).resolve().parents[1]


def _toy() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "Suburb": rng.choice(["Carlton", "Brunswick", "Box Hill"], n),
            "Type": rng.choice(["h", "u", "t"], n),
            "Rooms": rng.integers(1, 5, n),
            "Bathroom": rng.choice([1.0, 2.0, np.nan], n),
            "Car": rng.choice([0.0, 1.0, np.nan], n),
            "Landsize": rng.choice([300.0, 600.0, np.nan], n),
            "BuildingArea": rng.choice([90.0, 150.0, np.nan], n),
            "YearBuilt": rng.choice([1920.0, 1990.0, np.nan], n),
            "Distance": rng.uniform(2, 15, n),
            "Latitude": rng.uniform(-37.9, -37.7, n),
            "Longitude": rng.uniform(144.9, 145.1, n),
            "Price": rng.uniform(4e5, 2e6, n),
            "Date": pd.date_range("2016-02-01", periods=n, freq="W"),
        }
    )


def test_features_have_no_missing_values_and_fixed_order():
    df = _toy()
    X = transform(df, fit(df))
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()


def test_unseen_suburb_falls_back_instead_of_failing():
    df = _toy()
    spec = fit(df)
    unseen = df.head(1).assign(Suburb="Nowhere", Latitude=np.nan, Longitude=np.nan, Distance=np.nan)
    X = transform(unseen, spec)
    assert not X.isna().any().any()
    assert X["suburb_median_log"].iloc[0] == pytest.approx(spec.global_median_log)


def test_small_suburbs_are_shrunk_toward_the_prior():
    df = _toy()
    df.loc[df.index[0], "Suburb"] = "Tiny"
    df.loc[df.index[0], "Price"] = 9e6
    spec = fit(df)
    assert spec.suburb_median_log["Tiny"] < np.log(9e6)


@pytest.fixture(scope="module")
def metrics():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("run `python -m src.model.train` first")
    return json.loads(path.read_text())


def test_model_beats_the_median_baseline(metrics):
    model, base = metrics["model_xgboost"], metrics["baseline_suburb_type_median"]
    assert model["median_ape"] < base["median_ape"]
    assert model["mae"] < base["mae"]


def test_price_range_is_honest_on_future_sales(metrics):
    coverage = metrics["interval"]["coverage_on_test"]
    assert abs(coverage - metrics["interval"]["nominal"]) < 0.05


def test_split_is_by_time_not_random(metrics):
    assert metrics["train_period"][1] < metrics["test_period"][0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_model_matches_python():
    fixture = ROOT / "tests" / "parity_fixture.json"
    if not fixture.exists():
        pytest.skip("run `python -m src.model.train` first")
    out = subprocess.run(
        ["node", str(ROOT / "tests" / "parity.mjs")], capture_output=True, text=True, check=True
    )
    js = np.array(json.loads(out.stdout))
    py = np.array(json.loads(fixture.read_text())["y_log"])
    # in dollars this is well under a cent of difference on a $1m home
    assert np.max(np.abs(js - py)) < 1e-4


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_builds_the_same_features_as_python():
    meta = json.loads((ROOT / "web" / "spec.json").read_text())
    from src.features.build import FeatureSpec

    spec = FeatureSpec(**meta["spec"])
    suburb = meta["suburbs"][0]
    rows = [
        dict(Suburb=suburb, Type="h", Rooms=3, Bathroom=2.0, Car=1.0, Landsize=450.0,
             BuildingArea=160.0, YearBuilt=1995.0),
        dict(Suburb=suburb, Type="u", Rooms=2, Bathroom=np.nan, Car=np.nan, Landsize=np.nan,
             BuildingArea=np.nan, YearBuilt=np.nan),
    ]
    df = pd.DataFrame(rows).assign(Distance=np.nan, Latitude=np.nan, Longitude=np.nan,
                                   Date=pd.Timestamp("2018-03-10"))
    py = transform(df, spec).to_numpy()

    def js_input(r):
        clean = lambda v: None if (isinstance(v, float) and np.isnan(v)) else v
        return {"suburb": r["Suburb"], "type": r["Type"], "rooms": r["Rooms"], "bath": clean(r["Bathroom"]),
                "car": clean(r["Car"]), "land": clean(r["Landsize"]), "building": clean(r["BuildingArea"]),
                "year": clean(r["YearBuilt"])}

    cases = [{"input": js_input(r), "months": 26} for r in rows]
    out = subprocess.run(["node", str(ROOT / "tests" / "features_parity.mjs"), json.dumps(cases)],
                         capture_output=True, text=True, check=True)
    js = np.array(json.loads(out.stdout), dtype=float)
    assert np.allclose(js, py, atol=1e-9)
