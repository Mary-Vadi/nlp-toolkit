"""String-level cleaning functions.

Every function maps ``str -> str`` so they can be chained freely.
See ``pipeline.py`` for preset combinations.
"""

from __future__ import annotations

import html
import re
import string
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
REPEAT_CHAR_RE = re.compile(r"(\w)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0000FE0F"             # variation selector-16
    "\U0000200D"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# Typographic characters common in text scraped from the web, Word or PDFs.
SMART_CHARS = {
    chr(0x2018): "'", chr(0x2019): "'", chr(0x201A): "'", chr(0x201B): "'",
    chr(0x201C): '"', chr(0x201D): '"', chr(0x201E): '"', chr(0x201F): '"',
    chr(0x2013): "-", chr(0x2014): "-", chr(0x2212): "-",
    chr(0x2026): "...",
    chr(0x00A0): " ",  # no-break space
    chr(0x200B): "",   # zero-width space
}

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "shan't": "shall not",
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)),
    re.IGNORECASE,
)


# Normalisation (safe for transformer inputs)

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply Unicode normalisation (folds full-width forms, ligatures, etc.)."""
    return unicodedata.normalize(form, text)


def fix_smart_characters(text: str) -> str:
    """Map curly quotes, long dashes and special spaces to ASCII."""
    for src, dst in SMART_CHARS.items():
        text = text.replace(src, dst)
    return text


def unescape_html(text: str) -> str:
    """Decode HTML entities, e.g. ``&amp;`` -> ``&``."""
    return html.unescape(text)


def remove_html_tags(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def remove_control_characters(text: str) -> str:
    """Drop non-printing control characters. Newlines and tabs are kept."""
    return CONTROL_CHAR_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


# Removal / masking. Pass ``token`` to mask instead of delete.

def replace_urls(text: str, token: str = "") -> str:
    return URL_RE.sub(f" {token} " if token else " ", text)


def replace_emails(text: str, token: str = "") -> str:
    return EMAIL_RE.sub(f" {token} " if token else " ", text)


def replace_mentions(text: str, token: str = "") -> str:
    return MENTION_RE.sub(f" {token} " if token else " ", text)


def replace_numbers(text: str, token: str = "") -> str:
    return NUMBER_RE.sub(f" {token} " if token else " ", text)


def handle_hashtags(text: str, keep_word: bool = True) -> str:
    """Strip the ``#`` and keep the word, or drop the hashtag entirely."""
    return HASHTAG_RE.sub(r"\1" if keep_word else " ", text)


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)


def remove_punctuation(text: str, keep: str = "") -> str:
    """Replace ASCII punctuation with spaces, except characters in ``keep``."""
    drop = "".join(c for c in string.punctuation if c not in keep)
    return text.translate(str.maketrans(drop, " " * len(drop)))


def remove_accents(text: str) -> str:
    """Strip diacritics. Only appropriate where accents are not meaningful."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# Lexical changes. These alter the input distribution, so avoid them
# for pretrained transformers unless you have a reason.

def to_lowercase(text: str) -> str:
    return text.lower()


def expand_contractions(text: str) -> str:
    """Expand English contractions, e.g. ``don't`` -> ``do not``."""
    text = fix_smart_characters(text)
    return _CONTRACTION_RE.sub(lambda m: CONTRACTIONS.get(m.group(0).lower(), m.group(0)), text)


def reduce_repeated_characters(text: str, max_repeat: int = 2) -> str:
    """Cap runs of the same character, e.g. ``soooo`` -> ``soo``."""
    return REPEAT_CHAR_RE.sub(lambda m: m.group(1) * max_repeat, text)


__all__ = [
    "normalize_unicode", "fix_smart_characters", "unescape_html",
    "remove_html_tags", "remove_control_characters", "normalize_whitespace",
    "replace_urls", "replace_emails", "replace_mentions", "replace_numbers",
    "handle_hashtags", "remove_emojis", "remove_punctuation", "remove_accents",
    "to_lowercase", "expand_contractions", "reduce_repeated_characters",
]
