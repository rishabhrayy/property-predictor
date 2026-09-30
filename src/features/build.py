"""Feature engineering.

Every statistic here (suburb medians, suburb coordinates, fill values) is learned from the
training split only and then applied to test data, so nothing about the test period leaks in.
The fitted `FeatureSpec` is also exported to JSON so the browser demo builds features exactly
the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TYPES = ["h", "t", "u"]  # house, townhouse, unit
EPOCH = pd.Timestamp("2016-01-01")

# Order matters: the exported JavaScript model reads features by position.
FEATURES = [
    "rooms",
    "bathrooms",
    "car",
    "landsize_log",
    "building_area_log",
    "year_built",
    "type_h",
    "type_t",
    "type_u",
    "distance_km",
    "latitude",
    "longitude",
    "suburb_median_log",
    "suburb_type_median_log",
    "months_since_2016",
]


@dataclass
class FeatureSpec:
    """Everything learned from the training data that feature building depends on."""

    global_median_log: float
    type_median_log: dict[str, float]
    suburb_median_log: dict[str, float]
    suburb_type_median_log: dict[str, float]
    suburb_geo: dict[str, dict[str, float]]
    fill: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return self.__dict__


def _smoothed_median(group_log_prices: pd.Series, prior: float, k: int = 10) -> float:
    """Shrink small-sample suburb medians toward a prior, so a suburb with 3 sales is not trusted blindly."""
    n = len(group_log_prices)
    return float((n * group_log_prices.median() + k * prior) / (n + k))


def fit(train: pd.DataFrame) -> FeatureSpec:
    log_price = np.log(train["Price"])
    global_median = float(log_price.median())
    type_median = {t: float(log_price[train["Type"] == t].median()) for t in TYPES}

    suburb_median = {
        s: _smoothed_median(lp, global_median) for s, lp in log_price.groupby(train["Suburb"])
    }
    suburb_type_median = {
        f"{s}|{t}": _smoothed_median(lp, suburb_median.get(s, type_median[t]))
        for (s, t), lp in log_price.groupby([train["Suburb"], train["Type"]])
    }
    geo = (
        train.groupby("Suburb")
        .agg(lat=("Latitude", "median"), lon=("Longitude", "median"), dist=("Distance", "median"))
        .dropna()
    )
    suburb_geo = {s: {k: float(v) for k, v in row.items()} for s, row in geo.iterrows()}

    fill = {
        "bathrooms": float(train["Bathroom"].median()),
        "car": float(train["Car"].median()),
        "landsize_log": float(np.log1p(train["Landsize"]).median()),
        "building_area_log": float(np.log1p(train["BuildingArea"]).median()),
        "year_built": float(train["YearBuilt"].median()),
        "latitude": float(train["Latitude"].median()),
        "longitude": float(train["Longitude"].median()),
        "distance_km": float(train["Distance"].median()),
    }
    return FeatureSpec(global_median, type_median, suburb_median, suburb_type_median, suburb_geo, fill)


def transform(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build the model matrix. Missing inputs are filled deterministically, never left as NaN."""
    geo = df["Suburb"].map(spec.suburb_geo)
    out = pd.DataFrame(index=df.index)
    out["rooms"] = df["Rooms"].astype(float)
    out["bathrooms"] = df["Bathroom"].astype(float).fillna(spec.fill["bathrooms"])
    out["car"] = df["Car"].astype(float).fillna(spec.fill["car"])
    out["landsize_log"] = np.log1p(df["Landsize"].astype(float)).fillna(spec.fill["landsize_log"])
    out["building_area_log"] = np.log1p(df["BuildingArea"].astype(float)).fillna(
        spec.fill["building_area_log"]
    )
    out["year_built"] = df["YearBuilt"].astype(float).fillna(spec.fill["year_built"])
    for t in TYPES:
        out[f"type_{t}"] = (df["Type"] == t).astype(float)
    out["distance_km"] = df["Distance"].astype(float).fillna(
        geo.map(lambda g: g["dist"] if isinstance(g, dict) else np.nan)
    ).fillna(spec.fill["distance_km"])
    out["latitude"] = df["Latitude"].astype(float).fillna(
        geo.map(lambda g: g["lat"] if isinstance(g, dict) else np.nan)
    ).fillna(spec.fill["latitude"])
    out["longitude"] = df["Longitude"].astype(float).fillna(
        geo.map(lambda g: g["lon"] if isinstance(g, dict) else np.nan)
    ).fillna(spec.fill["longitude"])
    out["suburb_median_log"] = df["Suburb"].map(spec.suburb_median_log).fillna(spec.global_median_log)
    type_prior = df["Type"].map(spec.type_median_log)
    out["suburb_type_median_log"] = (
        (df["Suburb"] + "|" + df["Type"]).map(spec.suburb_type_median_log).fillna(out["suburb_median_log"])
    ).fillna(type_prior)
    out["months_since_2016"] = (
        (df["Date"].dt.year - EPOCH.year) * 12 + (df["Date"].dt.month - EPOCH.month)
    ).astype(float)
    return out[FEATURES]
