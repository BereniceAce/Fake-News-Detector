import re
from spellchecker import SpellChecker

FAKE_INDICATORS = [
    "shocking", "you won't believe", "miracle cure", "secretly",
    "banned", "exposed", "leaked", "breaking", "urgent",
    "conspiracy", "win a million dollars", "chance to win",
    "god", "terrifying", "heartbreaking", "furious", "unbelievable outrage"
]

REAL_INDICATORS = [
    "study", "researchers", "confirmed", "evidence",
    "according to", "official report", "peer reviewed", "IEE",
    "APA", "sources", "references"
]

NONVALID_URLS = [".com", "cust-login.ie", "http", "-"]
VALID_URLS = [".edu", ".gov", "https", ".org"]

def sanitize_text(text):
    if text is None: return ""
    text = str(text).lower()
    text = re.sub(r"^\s*|\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([.!?])\s*", r"\1", text)
    return text

def keyword_pts(text, keywords):
    return sum(1 for kw in keywords if kw in text)

def punctuation_pts(text):
    bad_pts = 0
    if text.count("?") > 1: bad_pts += 1
    if text.count("!") > 1: bad_pts += 1
    if text.isupper(): bad_pts += 1
    return 0, bad_pts

spellCheck = SpellChecker()

def conciseness(text):
    #spellCheck = SpellChecker()
    words = text.split()
    misspelled = spellCheck.unknown(words)
    return 0, len(misspelled)

def url_pts(url):
    good_pts, bad_pts = 0, 0
    for bad in NONVALID_URLS:
        if bad in url: bad_pts += 1
    for good in VALID_URLS:
        if good in url: good_pts += 1
    return good_pts, bad_pts

def extract_features(text, url=""):
    text = sanitize_text(text)
    fake_kw = keyword_pts(text, FAKE_INDICATORS)
    real_kw = keyword_pts(text, REAL_INDICATORS)
    _, punct_bad = punctuation_pts(text)
    _, spell_bad = conciseness(text)
    url_good, url_bad = url_pts(url)
    return {
        "fake_kw": fake_kw,
        "real_kw": real_kw,
        "punct_bad": punct_bad,
        "spell_bad": spell_bad,
        "url_good": url_good,
        "url_bad": url_bad
    }
