"""Composable cleaning pipelines and presets.

Presets:

``transformer``
    Light normalisation for pretrained encoders (BERT, RoBERTa, ModernBERT).
    Case, punctuation and stopwords are kept because the tokenizer uses them.
``classic``
    Aggressive normalisation for sparse models (TF-IDF, BoW, Naive Bayes).
``social``
    Tweets and forum posts. URLs and handles are masked, hashtags unwrapped.

>>> from nlp_toolkit import CleaningPipeline
>>> pipe = CleaningPipeline.from_preset("classic")
>>> pipe("Check THIS out!!! https://x.com &amp; it's sooooo great")
'check this out it is soo great'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterable

from . import cleaning as C

Step = Callable[[str], str]


def _name(step: Step) -> str:
    if isinstance(step, partial):
        args = ", ".join(f"{k}={v!r}" for k, v in step.keywords.items())
        return f"{step.func.__name__}({args})"
    return getattr(step, "__name__", repr(step))


_BASE = [
    C.unescape_html,
    C.remove_html_tags,
    C.normalize_unicode,
    C.fix_smart_characters,
    C.remove_control_characters,
]

PRESETS: dict[str, list[Step]] = {
    "transformer": _BASE + [
        partial(C.replace_urls, token="[URL]"),
        partial(C.replace_emails, token="[EMAIL]"),
        C.normalize_whitespace,
    ],
    "classic": _BASE + [
        C.replace_urls,
        C.replace_emails,
        C.replace_mentions,
        C.handle_hashtags,
        C.remove_emojis,
        C.expand_contractions,
        C.to_lowercase,
        C.replace_numbers,
        C.remove_punctuation,
        C.reduce_repeated_characters,
        C.normalize_whitespace,
    ],
    "social": _BASE + [
        partial(C.replace_urls, token="<URL>"),
        partial(C.replace_mentions, token="<USER>"),
        C.handle_hashtags,
        C.remove_emojis,
        C.reduce_repeated_characters,
        C.normalize_whitespace,
    ],
}


@dataclass
class CleaningPipeline:
    """Ordered sequence of ``str -> str`` steps, applied left to right."""

    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_preset(cls, name: str) -> "CleaningPipeline":
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; expected one of {list(PRESETS)}")
        return cls(list(PRESETS[name]))

    def add(self, step: Step) -> "CleaningPipeline":
        self.steps.append(step)
        return self

    def remove(self, step_name: str) -> "CleaningPipeline":
        """Drop all steps whose name starts with ``step_name``."""
        self.steps = [s for s in self.steps if not _name(s).startswith(step_name)]
        return self

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            return ""  # None / NaN
        for step in self.steps:
            text = step(text)
        return text

    def clean_many(self, texts: Iterable[str]) -> list[str]:
        return [self(t) for t in texts]

    def trace(self, text: str) -> list[tuple[str, str]]:
        """Return ``(step_name, text)`` after each step, for debugging."""
        out = [("input", text)]
        for step in self.steps:
            text = step(text)
            out.append((_name(step), text))
        return out

    def describe(self) -> str:
        return "\n".join(f"{i:>2}. {_name(s)}" for i, s in enumerate(self.steps, 1))

    def __repr__(self) -> str:
        return f"CleaningPipeline({len(self.steps)} steps)"
