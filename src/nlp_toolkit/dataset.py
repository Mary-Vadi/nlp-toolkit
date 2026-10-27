"""DataFrame-level utilities for building text classification datasets.

Suggested order:
    load -> drop_empty -> clean_column -> drop_label_conflicts
    -> remove_duplicates -> filter_by_length -> split_dataset -> check_leakage
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

import pandas as pd

_NORM_RE = re.compile(r"[\W_]+")

_READERS = {
    ".csv": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}


def load_text_data(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return reader(path, **kwargs)


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path, fmt: str = "csv") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        if fmt == "csv":
            df.to_csv(out / f"{name}.csv", index=False)
        elif fmt == "jsonl":
            df.to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        elif fmt == "parquet":
            df.to_parquet(out / f"{name}.parquet", index=False)
        else:
            raise ValueError("fmt must be 'csv', 'jsonl' or 'parquet'")


def _fingerprint(text: str) -> str:
    """Hash of the text with case, whitespace and punctuation removed."""
    norm = _NORM_RE.sub(" ", str(text).lower()).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def _log(fn: str, removed: int, total: int) -> None:
    print(f"[{fn}] removed {removed} of {total} rows")


def drop_empty(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    mask = df[text_col].notna() & df[text_col].astype(str).str.strip().ne("")
    return df.loc[mask].reset_index(drop=True)


def clean_column(
    df: pd.DataFrame,
    cleaner: Callable[[str], str],
    text_col: str = "text",
    new_col: str | None = None,
) -> pd.DataFrame:
    """Apply ``cleaner`` to ``text_col``. Writes to ``new_col`` if given."""
    df = df.copy()
    df[new_col or text_col] = df[text_col].map(cleaner)
    return df


def remove_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    mode: str = "normalized",
    report: bool = True,
) -> pd.DataFrame:
    """Keep the first occurrence of each text.

    ``mode="exact"`` compares raw strings; ``mode="normalized"`` ignores
    case, whitespace and punctuation.
    """
    if mode == "exact":
        keys = df[text_col]
    elif mode == "normalized":
        keys = df[text_col].map(_fingerprint)
    else:
        raise ValueError("mode must be 'exact' or 'normalized'")
    keep = ~keys.duplicated(keep="first")
    if report:
        _log("remove_duplicates", int((~keep).sum()), len(df))
    return df.loc[keep].reset_index(drop=True)


def find_label_conflicts(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Rows whose (normalised) text appears with more than one label."""
    keys = df[text_col].map(_fingerprint)
    n_labels = df.groupby(keys)[label_col].transform("nunique")
    return df.loc[n_labels > 1].assign(_group=keys[n_labels > 1]).sort_values("_group")


def drop_label_conflicts(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label", report: bool = True
) -> pd.DataFrame:
    """Drop every copy of a text that has conflicting labels.

    Call before ``remove_duplicates``, otherwise one arbitrary copy survives.
    """
    keys = df[text_col].map(_fingerprint)
    keep = df.groupby(keys)[label_col].transform("nunique") <= 1
    if report:
        _log("drop_label_conflicts", int((~keep).sum()), len(df))
    return df.loc[keep].reset_index(drop=True)


def filter_by_length(
    df: pd.DataFrame,
    text_col: str = "text",
    min_words: int = 1,
    max_words: int | None = None,
    report: bool = True,
) -> pd.DataFrame:
    n = df[text_col].astype(str).str.split().str.len()
    mask = n >= min_words
    if max_words is not None:
        mask &= n <= max_words
    if report:
        _log("filter_by_length", int((~mask).sum()), len(df))
    return df.loc[mask].reset_index(drop=True)


def detect_language(df: pd.DataFrame, text_col: str = "text", new_col: str = "lang") -> pd.DataFrame:
    """Add an ISO language code column. Requires ``langdetect``."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0

    def _detect(t: str) -> str:
        try:
            return detect(t)
        except LangDetectException:
            return "unknown"

    df = df.copy()
    df[new_col] = df[text_col].astype(str).map(_detect)
    return df


def check_labels(df: pd.DataFrame, label_col: str = "label", warn_ratio: float = 3.0) -> pd.DataFrame:
    counts = df[label_col].value_counts(dropna=False)
    table = pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(2)})
    ratio = counts.max() / max(counts.min(), 1)
    if ratio > warn_ratio:
        print(f"[check_labels] class imbalance: max/min = {ratio:.1f}")
    return table


def split_dataset(
    df: pd.DataFrame,
    label_col: str | None = "label",
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Train/val/test split, stratified on ``label_col`` if given."""
    from sklearn.model_selection import train_test_split

    strat = df[label_col] if label_col else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    strat = train_val[label_col] if label_col else None
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size), random_state=seed, stratify=strat
    )
    return {k: v.reset_index(drop=True) for k, v in {"train": train, "val": val, "test": test}.items()}


def check_leakage(splits: dict[str, pd.DataFrame], text_col: str = "text") -> dict[str, int]:
    """Count normalised texts shared between each pair of splits."""
    prints = {k: set(v[text_col].map(_fingerprint)) for k, v in splits.items()}
    names = list(prints)
    return {
        f"{a}-{b}": len(prints[a] & prints[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
