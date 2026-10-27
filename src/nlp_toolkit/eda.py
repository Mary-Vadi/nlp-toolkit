"""Descriptive statistics and plots for text datasets."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_WORD_RE = re.compile(r"\b\w+\b")
_SENT_RE = re.compile(r"[.!?]+")


def _ttr(text: str) -> float:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return len(set(words)) / len(words) if words else 0.0


def text_stats(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add per-row length and lexical statistics."""
    s = df[text_col].astype(str)
    out = df.copy()
    out["n_chars"] = s.str.len()
    out["n_words"] = s.map(lambda t: len(_WORD_RE.findall(t)))
    out["n_sentences"] = s.map(lambda t: max(1, sum(1 for x in _SENT_RE.split(t) if x.strip())))
    out["avg_word_len"] = (out["n_chars"] / out["n_words"].clip(lower=1)).round(2)
    out["upper_ratio"] = s.map(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)).round(3)
    out["type_token_ratio"] = s.map(_ttr).round(3)
    return out


def summary(df: pd.DataFrame, text_col: str = "text", label_col: str | None = "label") -> pd.DataFrame:
    stats = text_stats(df, text_col)
    cols = ["n_chars", "n_words", "n_sentences", "avg_word_len", "type_token_ratio"]
    if label_col and label_col in df:
        return stats.groupby(label_col)[cols].mean().round(2)
    return stats[cols].describe().round(2)


def top_ngrams(texts, n: int = 1, k: int = 20, lowercase: bool = True) -> pd.DataFrame:
    counter: Counter = Counter()
    for t in texts:
        words = _WORD_RE.findall(t.lower() if lowercase else t)
        counter.update(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return pd.DataFrame(counter.most_common(k), columns=["ngram", "count"])


def vocab_size(texts, lowercase: bool = True) -> int:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(_WORD_RE.findall(t.lower() if lowercase else t))
    return len(vocab)


def plot_length_distribution(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str | None = "label",
    unit: str = "n_words",
    bins: int = 40,
):
    import matplotlib.pyplot as plt

    stats = text_stats(df, text_col)
    fig, ax = plt.subplots(figsize=(8, 4))
    if label_col and label_col in df:
        for label, group in stats.groupby(label_col):
            ax.hist(group[unit], bins=bins, alpha=0.6, label=str(label))
        ax.legend(title=label_col)
    else:
        ax.hist(stats[unit], bins=bins)
    ax.set_xlabel(unit)
    ax.set_ylabel("count")
    ax.set_title("Length distribution")
    fig.tight_layout()
    return fig


def plot_top_ngrams(texts, n: int = 1, k: int = 15, title: str | None = None):
    import matplotlib.pyplot as plt

    top = top_ngrams(texts, n=n, k=k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * k + 1))
    ax.barh(top["ngram"], top["count"])
    ax.set_xlabel("count")
    ax.set_title(title or f"Top {k} {n}-grams")
    fig.tight_layout()
    return fig
