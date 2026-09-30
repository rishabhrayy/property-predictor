"""Load and clean the Melbourne Housing Market dataset.

Source: Tony Pino, "Melbourne Housing Market" on Kaggle (scraped from Domain.com.au,
2016-2018), licensed CC BY-NC-SA 4.0. Downloaded with kagglehub so the raw file is never
committed; see README for attribution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET = "anthonypino/melbourne-housing-market"
FILENAME = "Melbourne_housing_FULL.csv"

# Sales outside this range are almost always data-entry errors or non-residential lots.
MIN_PRICE = 100_000
MAX_PRICE = 8_000_000


def download() -> Path:
    """Fetch the dataset (cached by kagglehub after the first call) and return the CSV path."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET)) / FILENAME


def load(path: Path | None = None) -> pd.DataFrame:
    """Return one row per priced sale, with parsed dates and obviously bad rows removed."""
    df = pd.read_csv(path or download())
    df = df.rename(columns={"Lattitude": "Latitude", "Longtitude": "Longitude"})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df = df[df["Price"].between(MIN_PRICE, MAX_PRICE)]
    df = df[df["Type"].isin(["h", "u", "t"])]
    df = df[df["Rooms"].between(1, 10)]

    # Physically impossible or clearly mistyped values become missing, not outliers the model learns.
    df.loc[~df["Landsize"].between(1, 20_000), "Landsize"] = pd.NA
    df.loc[~df["BuildingArea"].between(20, 1_500), "BuildingArea"] = pd.NA
    df.loc[~df["YearBuilt"].between(1840, 2018), "YearBuilt"] = pd.NA
    df.loc[df["Bathroom"] > 8, "Bathroom"] = pd.NA
    df.loc[df["Car"] > 10, "Car"] = pd.NA

    return df.sort_values("Date").reset_index(drop=True)
