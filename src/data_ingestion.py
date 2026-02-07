from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

import yaml

LOGGER = logging.getLogger(__name__)


def load_params(path: str | os.PathLike = "params.yml") -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_source_df(source_url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(source_url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read CSV from {source_url!r}. Check internet access and the URL."
        ) from e


def preprocess_df(
    df: pd.DataFrame,
    allowed_sentiments: list[str],
    label_map: dict[str, int],
) -> pd.DataFrame:
    required_cols = {"sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    df = df.drop(columns=["tweet_id"], errors="ignore")
    filtered = df[df["sentiment"].isin(allowed_sentiments)].copy()
    if filtered.empty:
        raise ValueError(
            "No rows left after filtering sentiments. "
            f"allowed_sentiments={allowed_sentiments}"
        )

    filtered["sentiment"] = filtered["sentiment"].replace(label_map)
    return filtered


def split_train_test(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["sentiment"],
        )
    except Exception:
        # Fallback if stratify fails for any reason (e.g., tiny class counts).
        return train_test_split(df, test_size=test_size, random_state=random_state)


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | os.PathLike,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.csv"
    test_path = out_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    LOGGER.info("Wrote %s (%d rows) and %s (%d rows)", train_path, len(train_df), test_path, len(test_df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = load_params()
    ingestion = params.get("data_ingestion") or {}

    source_url = str(ingestion.get("source_url") or "")
    if not source_url:
        raise ValueError("params.yml missing: data_ingestion.source_url")

    allowed_sentiments = ingestion.get("allowed_sentiments") or ["sadness", "happiness"]
    if not isinstance(allowed_sentiments, list) or not all(
        isinstance(x, str) for x in allowed_sentiments
    ):
        raise ValueError("params.yml invalid: data_ingestion.allowed_sentiments must be a list of strings")

    label_map = ingestion.get("label_map") or {"sadness": 0, "happiness": 1}
    if not isinstance(label_map, dict):
        raise ValueError("params.yml invalid: data_ingestion.label_map must be a mapping")

    test_size = float(ingestion.get("test_size", 0.2))
    random_state = int(ingestion.get("random_state", 42))
    out_dir = str(ingestion.get("out_dir", "data/raw"))

    df = load_source_df(source_url)
    final_df = preprocess_df(df, allowed_sentiments=allowed_sentiments, label_map=label_map)
    train_df, test_df = split_train_test(final_df, test_size=test_size, random_state=random_state)
    save_splits(train_df, test_df, out_dir=out_dir)


if __name__ == "__main__":
    main()
