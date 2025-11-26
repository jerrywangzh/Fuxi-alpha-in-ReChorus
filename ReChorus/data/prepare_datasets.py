#!/usr/bin/env python3
"""Utility to download and format the built-in ReChorus datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import urllib.request
from urllib.error import HTTPError

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
NEG_SAMPLE_SIZE = 100


def download_file(url: str, target: Path, force: bool = False) -> None:
    """Download ``url`` into ``target`` if the file does not exist."""
    if target.exists() and not force:
        print(f"[download] Reuse existing file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {url} -> {target}")
    try:
        with urllib.request.urlopen(url) as resp, open(target, "wb") as fout:
            shutil.copyfileobj(resp, fout)
    except HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {url} (HTTP {exc.code}). "
            "Please download the file manually and place it at the target path."
        ) from exc


def extract_zip(zip_path: Path, expected_dir: Path, force: bool = False) -> Path:
    """
    Extract ``zip_path`` so that ``expected_dir`` exists; return the directory path.
    """
    if expected_dir.exists() and not force:
        print(f"[extract] Reuse extracted folder: {expected_dir}")
        return expected_dir
    print(f"[extract] Unzipping {zip_path} -> {expected_dir.parent}")
    expected_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(expected_dir.parent)
    return expected_dir


def sample_negatives(all_items: Sequence[int], user_items: set[int], rng: np.random.Generator, k: int) -> List[int]:
    """Sample ``k`` negatives for a user from the global item universe."""
    pool = [item for item in all_items if item not in user_items]
    if not pool:
        return []
    size = min(k, len(pool))
    indices = rng.choice(len(pool), size=size, replace=False)
    return [pool[int(i)] for i in indices]


def build_topk_files(df: pd.DataFrame, out_dir: Path, neg_size: int, seed: int = 42) -> None:
    """Split chronological interactions into train/dev/test csv files."""
    rng = np.random.default_rng(seed)
    all_items = sorted(df["item_id"].unique().tolist())
    train_rows: List[Dict[str, int]] = []
    dev_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("time")
        if len(group) < 3:
            continue
        user_items = set(group["item_id"].tolist())
        test_row = group.iloc[-1]
        dev_row = group.iloc[-2]
        train_part = group.iloc[:-2]
        train_rows.extend(train_part[["user_id", "item_id", "time"]].to_dict("records"))
        neg_dev = sample_negatives(all_items, user_items, rng, neg_size)
        neg_test = sample_negatives(all_items, user_items, rng, neg_size)
        dev_rows.append(
            {
                "user_id": int(dev_row.user_id),
                "item_id": int(dev_row.item_id),
                "time": int(dev_row.time),
                "neg_items": json.dumps(neg_dev),
            }
        )
        test_rows.append(
            {
                "user_id": int(test_row.user_id),
                "item_id": int(test_row.item_id),
                "time": int(test_row.time),
                "neg_items": json.dumps(neg_test),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_csv(out_dir / "train.csv", sep="\t", index=False)
    pd.DataFrame(dev_rows).to_csv(out_dir / "dev.csv", sep="\t", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "test.csv", sep="\t", index=False)
    print(f"[split] Saved train/dev/test to {out_dir}")


def prepare_grocery(force: bool = False) -> None:
    """The Amazon Grocery dataset ships with ReChorus; verify its presence."""
    dataset_dir = DATA_ROOT / "Grocery_and_Gourmet_Food"
    required = [dataset_dir / name for name in ("train.csv", "dev.csv", "test.csv", "item_meta.csv")]
    if all(path.exists() for path in required):
        print("[grocery] Files already prepared; no action required.")
    else:
        raise FileNotFoundError("[grocery] Missing built-in files. Run Amazon.ipynb to regenerate them.")


def prepare_movielens(force: bool = False, neg_size: int = NEG_SAMPLE_SIZE) -> None:
    dataset_dir = DATA_ROOT / "MovieLens_1M"
    if not force and all((dataset_dir / f).exists() for f in ("train.csv", "dev.csv", "test.csv")):
        print("[movielens] Existing csv files detected; skip generation.")
        return

    raw_dir = dataset_dir / "raw"
    zip_path = raw_dir / "ml-1m.zip"
    download_file("https://files.grouplens.org/datasets/movielens/ml-1m.zip", zip_path, force=force)
    extract_dir = raw_dir / "ml-1m"
    extract_zip(zip_path, extract_dir, force=force)
    ratings_path = extract_dir / "ratings.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.dat not found under {ratings_path}")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "time"],
        dtype={"user_id": np.int64, "item_id": np.int64, "rating": np.int64, "time": np.int64},
    )
    ratings = ratings.sort_values(["user_id", "time"])
    user_counts = ratings.groupby("user_id").size()
    keep_users = user_counts[user_counts >= 3].index
    ratings = ratings[ratings["user_id"].isin(keep_users)][["user_id", "item_id", "time"]]
    build_topk_files(ratings, dataset_dir, neg_size)


def parse_mind_behaviors(behaviors_path: Path) -> pd.DataFrame:
    """Convert the behavior log into (user,item,time) triples."""
    rows: List[Dict[str, object]] = []
    with open(behaviors_path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or not parts[4]:
                continue
            user_raw = parts[1]
            ts = int(datetime.strptime(parts[2], "%m/%d/%Y %I:%M:%S %p").timestamp())
            impressions = parts[4].split(" ")
            for entry in impressions:
                if not entry or "-" not in entry:
                    continue
                news_id, label = entry.split("-")
                if label == "1":
                    rows.append({"user_raw": user_raw, "item_raw": news_id, "time": ts})
    df = pd.DataFrame(rows)
    return df


def remap_ids(df: pd.DataFrame, col: str, target_col: str) -> None:
    """Map string identifiers to consecutive integer ids."""
    unique_vals = df[col].unique().tolist()
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    df[target_col] = df[col].map(mapping)


def prepare_mind(force: bool = False, neg_size: int = NEG_SAMPLE_SIZE) -> None:
    dataset_dir = DATA_ROOT / "MIND_Large"
    if not force and all((dataset_dir / f).exists() for f in ("train.csv", "dev.csv", "test.csv")):
        print("[mind] Existing csv files detected; skip generation.")
        return

    raw_dir = dataset_dir / "raw"
    train_zip = raw_dir / "MINDsmall_train.zip"
    download_file(
        "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        train_zip,
        force=force,
    )
    extract_dir = raw_dir / "MINDsmall_train"
    extract_zip(train_zip, extract_dir, force=force)
    behaviors_path = extract_dir / "behaviors.tsv"
    if not behaviors_path.exists():
        raise FileNotFoundError(f"behaviors.tsv not found under {behaviors_path}")

    interactions = parse_mind_behaviors(behaviors_path)
    if interactions.empty:
        raise RuntimeError("No positive interactions were parsed from MIND behaviors.")
    remap_ids(interactions, "user_raw", "user_id")
    remap_ids(interactions, "item_raw", "item_id")
    interactions = interactions[["user_id", "item_id", "time"]]
    build_topk_files(interactions, dataset_dir, neg_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReChorus datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "grocery", "movielens", "mind"],
        help="Datasets to prepare",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files")
    parser.add_argument("--neg_samples", type=int, default=NEG_SAMPLE_SIZE, help="#negative samples per dev/test item")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    targets = ["grocery", "movielens", "mind"] if "all" in args.datasets else args.datasets

    if "grocery" in targets:
        prepare_grocery(force=args.force)
    if "movielens" in targets:
        prepare_movielens(force=args.force, neg_size=args.neg_samples)
    if "mind" in targets:
        prepare_mind(force=args.force, neg_size=args.neg_samples)


if __name__ == "__main__":
    main()
