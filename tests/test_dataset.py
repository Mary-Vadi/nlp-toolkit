import pandas as pd

from nlp_toolkit import dataset as D
from nlp_toolkit import eda
from nlp_toolkit.preprocessing import ngrams, remove_stopwords, simple_tokenize


def make_df():
    return pd.DataFrame(
        {
            "text": ["Hello, World!", "hello world", "", None, "A longer text here.",
                     "Same text", "same TEXT", "one"] + [f"sample number {i}" for i in range(40)],
            "label": [0, 0, 1, 1, 1, 0, 1, 0] + [i % 2 for i in range(40)],
        }
    )


def test_drop_empty_and_duplicates():
    df = D.drop_empty(make_df())
    assert len(df) == 46
    df = D.remove_duplicates(df, report=False)
    assert len(df) == 44  # "hello world" and "same TEXT" removed


def test_label_conflicts():
    conflicts = D.find_label_conflicts(D.drop_empty(make_df()))
    assert set(conflicts["text"]) == {"Same text", "same TEXT"}


def test_drop_label_conflicts():
    df = D.drop_label_conflicts(D.drop_empty(make_df()), report=False)
    assert "Same text" not in set(df["text"]) and "same TEXT" not in set(df["text"])
    assert "Hello, World!" in set(df["text"])  # same label twice is not a conflict


def test_filter_by_length():
    df = D.filter_by_length(D.drop_empty(make_df()), min_words=2, report=False)
    assert "one" not in set(df["text"])


def test_split_and_leakage():
    df = D.remove_duplicates(D.drop_empty(make_df()), report=False)
    splits = D.split_dataset(df, val_size=0.2, test_size=0.2)
    assert sum(len(s) for s in splits.values()) == len(df)
    assert all(v == 0 for v in D.check_leakage(splits).values())


def test_eda():
    df = D.drop_empty(make_df())
    stats = eda.text_stats(df)
    assert {"n_words", "n_chars", "type_token_ratio"} <= set(stats.columns)
    top = eda.top_ngrams(df["text"], n=1, k=3)
    assert top.iloc[0]["ngram"] in {"sample", "number"}


def test_preprocessing_helpers():
    toks = simple_tokenize("This is not a test, it's real!")
    assert toks == ["This", "is", "not", "a", "test", "it's", "real"]
    assert "not" in remove_stopwords(toks)  # negations kept
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
