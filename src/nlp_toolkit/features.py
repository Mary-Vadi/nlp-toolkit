"""TF-IDF features and a logistic regression baseline."""

from __future__ import annotations

import pandas as pd


def tfidf_vectorizer(max_features: int = 50_000, ngram_range=(1, 2), **kwargs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        **kwargs,
    )


def baseline_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    ngram_range=(1, 2),
):
    """Fit TF-IDF + LogisticRegression, print a report, return (model, report_dict)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        tfidf_vectorizer(ngram_range=ngram_range),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(train[text_col], train[label_col])
    preds = model.predict(test[text_col])
    print(classification_report(test[label_col], preds, digits=4, zero_division=0))
    return model, classification_report(test[label_col], preds, output_dict=True, zero_division=0)


def top_features(model, k: int = 15) -> pd.DataFrame:
    """Highest-weighted n-grams for each class (binary classifiers only)."""
    vec, clf = model.steps[0][1], model.steps[-1][1]
    names = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    return pd.DataFrame(
        {
            f"top_{clf.classes_[0]}": names[order[:k]],
            f"top_{clf.classes_[1]}": names[order[-k:][::-1]],
        }
    )
