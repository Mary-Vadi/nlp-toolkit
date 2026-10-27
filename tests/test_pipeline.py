import math

import pytest

from nlp_toolkit import CleaningPipeline
from nlp_toolkit.cleaning import to_lowercase

EMOJI = chr(0x1F60D)
RAW = f"Check THIS out!!! https://x.com &amp; it's sooooo great {EMOJI} @maryam #NLP <br> 2026"


def test_transformer_preset_is_light():
    out = CleaningPipeline.from_preset("transformer")(RAW)
    assert "THIS" in out and "!!!" in out
    assert "[URL]" in out and "<br>" not in out and "&amp;" not in out


def test_classic_preset():
    assert CleaningPipeline.from_preset("classic")(RAW) == "check this out it is soo great nlp"


def test_social_preset():
    out = CleaningPipeline.from_preset("social")(RAW)
    assert "<URL>" in out and "<USER>" in out and "NLP" in out and EMOJI not in out


def test_unknown_preset():
    with pytest.raises(ValueError):
        CleaningPipeline.from_preset("nope")


def test_add_remove_and_trace():
    pipe = CleaningPipeline.from_preset("transformer").add(to_lowercase)
    assert pipe("ABC") == "abc"
    pipe.remove("to_lowercase")
    assert pipe("ABC") == "ABC"
    trace = pipe.trace("A  B")
    assert trace[0] == ("input", "A  B") and trace[-1][1] == "A B"


def test_missing_values():
    pipe = CleaningPipeline.from_preset("classic")
    assert pipe(None) == "" and pipe(math.nan) == ""
