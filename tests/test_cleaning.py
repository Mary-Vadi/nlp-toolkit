from nlp_toolkit import cleaning as C

EMOJI = chr(0x1F60D)
LQUOTE, RQUOTE, APOS, EM_DASH = chr(0x201C), chr(0x201D), chr(0x2019), chr(0x2014)


def test_urls_and_emails():
    t = "see https://example.com or www.test.org, mail me@uni.ac.uk"
    out = C.normalize_whitespace(C.replace_emails(C.replace_urls(t, "<URL>"), "<EMAIL>"))
    assert out == "see <URL> or <URL> mail <EMAIL>"


def test_html():
    assert C.normalize_whitespace(C.remove_html_tags(C.unescape_html("a&amp;b<br/>c"))) == "a&b c"


def test_smart_characters():
    raw = f"{LQUOTE}Hi{RQUOTE} {EM_DASH} it{APOS}s"
    assert C.fix_smart_characters(raw) == "\"Hi\" - it's"


def test_unicode_normalisation():
    full_width_ab = chr(0xFF21) + chr(0xFF22)
    assert C.normalize_unicode(full_width_ab) == "AB"


def test_mentions_hashtags_emojis():
    t = f"@maryam loves #NLP {EMOJI}"
    assert C.normalize_whitespace(C.replace_mentions(t)) == f"loves #NLP {EMOJI}"
    assert C.handle_hashtags(t) == f"@maryam loves NLP {EMOJI}"
    assert C.normalize_whitespace(C.remove_emojis(t)) == "@maryam loves #NLP"


def test_contractions():
    assert C.expand_contractions("I can't, she won't, they're") == "I cannot, she will not, they are"


def test_repeated_characters_and_numbers():
    assert C.reduce_repeated_characters("soooo goooood") == "soo good"
    out = C.normalize_whitespace(C.replace_numbers("in 2026 it cost 3.50", "<NUM>"))
    assert out == "in <NUM> it cost <NUM>"


def test_punctuation_keep():
    assert C.normalize_whitespace(C.remove_punctuation("hi! it's me.", keep="'")) == "hi it's me"


def test_accents():
    assert C.remove_accents("caf" + chr(0xE9)) == "cafe"
