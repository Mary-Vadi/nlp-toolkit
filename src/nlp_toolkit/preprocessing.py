"""Token-level preprocessing for sparse / classical models.

Not intended for transformer inputs, which use their own tokenizer.
NLTK resources are downloaded lazily on first use.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")

# Used when the NLTK corpus is unavailable (e.g. no network access).
_FALLBACK_EN = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself
yourselves
""".split())


def simple_tokenize(text: str, keep_punct: bool = False) -> list[str]:
    """Regex tokenizer with no external dependencies."""
    tokens = _TOKEN_RE.findall(text)
    if not keep_punct:
        tokens = [t for t in tokens if t[0].isalnum()]
    return tokens


def _ensure_nltk(path: str, package: str) -> None:
    import nltk

    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)


def nltk_tokenize(text: str) -> list[str]:
    import nltk

    _ensure_nltk("tokenizers/punkt_tab", "punkt_tab")
    return nltk.word_tokenize(text)


@lru_cache(maxsize=None)
def get_stopwords(language: str = "english") -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        _ensure_nltk("corpora/stopwords", "stopwords")
        return frozenset(stopwords.words(language))
    except LookupError:
        if language == "english":
            return _FALLBACK_EN
        raise


def remove_stopwords(
    tokens: Iterable[str],
    language: str = "english",
    extra: Iterable[str] = (),
    keep: Iterable[str] = ("not", "no", "nor"),
) -> list[str]:
    """Filter stopwords. Negations are kept by default since they carry polarity."""
    stop = (set(get_stopwords(language)) | set(extra)) - set(keep)
    return [t for t in tokens if t.lower() not in stop]


def stem(tokens: Iterable[str], language: str = "english") -> list[str]:
    from nltk.stem import SnowballStemmer

    stemmer = SnowballStemmer(language)
    return [stemmer.stem(t) for t in tokens]


def lemmatize(tokens: Iterable[str]) -> list[str]:
    from nltk.stem import WordNetLemmatizer

    _ensure_nltk("corpora/wordnet", "wordnet")
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def ngrams(tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def preprocess(
    text: str,
    stopwords: bool = True,
    method: str | None = "lemma",
    language: str = "english",
) -> list[str]:
    """Tokenize, optionally drop stopwords, then lemmatize or stem.

    ``method`` is one of ``"lemma"``, ``"stem"`` or ``None``.
    """
    tokens = simple_tokenize(text)
    if stopwords:
        tokens = remove_stopwords(tokens, language=language)
    if method == "lemma":
        tokens = lemmatize(tokens)
    elif method == "stem":
        tokens = stem(tokens, language=language)
    return tokens
