"""
preprocessing.py
-----------------
Shared text-cleaning and preprocessing utilities used by both the training
script (train.py) and the Flask web app (app.py), so that emails are
processed identically at training time and at prediction time.

Pipeline (per the dissertation methodology, Chapter 3):
 1. Combine subject + body
 2. Clean text: lower-case, strip HTML, normalise URLs/emails, remove
    non-alphabetic characters
 3. Tokenise (simple regex word tokeniser -- no external corpus downloads
    required, so this runs anywhere without extra setup)
 4. Remove English stop words (scikit-learn's built-in list -- no download
    required)
 5. Stem remaining tokens with the Porter stemmer (rule-based, no corpus
    download required) as a lightweight stand-in for full lemmatisation
 6. Re-join into a cleaned string ready for TF-IDF vectorisation
"""

import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
_stopwords = set(ENGLISH_STOP_WORDS)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def clean_text(subject: str, body: str) -> str:
    """Combine subject + body and clean into a normalised string of
    stemmed tokens, ready for TF-IDF vectorisation."""
    subject = subject or ""
    body = body or ""
    text = f"{subject} {body}"

    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" urllink ", text)
    text = _EMAIL_RE.sub(" emailaddr ", text)
    text = text.lower()

    tokens = _TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in _stopwords]
    tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def extract_stylometric_features(subject: str, body: str) -> dict:
    """A small supplementary stylometric feature set, informed by the
    dissertation's literature review (Opara, Modesti & Golightly, 2025):
    imperative-verb-style cues, first-person pronoun usage, punctuation-based
    urgency indicators, and basic length/diversity statistics.
    These are returned for optional analysis/display; the core model uses
    the TF-IDF representation from clean_text() above.
    """
    subject = subject or ""
    body = body or ""
    text = f"{subject} {body}"
    lower = text.lower()
    words = re.findall(r"[a-zA-Z']+", lower)
    n_words = max(len(words), 1)

    urgency_words = ["urgent", "immediately", "verify", "suspend", "click",
                      "confirm", "password", "act now", "limited time",
                      "expire", "alert", "unauthorized", "unauthorised"]
    first_person = ["i", "we", "our", "us"]

    return {
        "word_count": n_words,
        "exclamation_count": text.count("!"),
        "url_count": len(_URL_RE.findall(text)),
        "urgency_word_count": sum(lower.count(w) for w in urgency_words),
        "first_person_ratio": round(sum(words.count(w) for w in first_person) / n_words, 4),
        "lexical_diversity": round(len(set(words)) / n_words, 4),
    }


# ---------------------------------------------------------------------------
# Heuristic "AI-generated writing style" indicator
# ---------------------------------------------------------------------------
# IMPORTANT: this is a transparent, rule-based heuristic built from stylistic
# markers reported in the literature (Chen et al., 2026; Opara, Modesti &
# Golightly, 2025; general human-vs-LLM stylometry research) -- it is NOT a
# trained/validated machine learning classifier. The dataset used elsewhere
# in this project contains no real human-written emails, so a genuine
# supervised "human vs AI" classifier could not be trained or validated here.
# This score should be presented to users as an indicative, exploratory
# signal only, never as a confident prediction.

_STOCK_PHRASES = [
    "i hope this email finds you well", "please don't hesitate to reach out",
    "please do not hesitate to contact", "feel free to reach out",
    "thank you for your understanding", "i look forward to hearing from you",
    "please let me know if you have any questions", "should you have any questions",
    "at your earliest convenience", "in light of", "furthermore", "moreover",
    "additionally", "in conclusion", "it is important to note that",
    "i would like to inform you", "we appreciate your understanding",
    "please rest assured", "warm regards", "kind regards", "best regards",
]
_CONTRACTIONS = [
    "don't", "can't", "won't", "i'm", "it's", "that's", "you're", "we're",
    "didn't", "isn't", "doesn't", "wouldn't", "couldn't", "shouldn't",
    "i've", "you've", "we've", "they've", "i'll", "you'll", "we'll",
]
_ELONGATION_RE = re.compile(r"([a-zA-Z])\1{2,}")  # e.g. "soooo", "hiiii"
_SHOUTY_RE = re.compile(r"\b[A-Z]{3,}\b")


def estimate_ai_writing_style(subject: str, body: str) -> dict:
    """Return a 0-100 heuristic score indicating how strongly this email's
    *writing style* matches patterns associated with AI-generated text in
    the literature, plus a breakdown of which signals fired.

    This is explicitly a heuristic, not a trained classifier -- see the
    module-level note above.
    """
    subject = subject or ""
    body = body or ""
    text = f"{subject} {body}".strip()
    lower = text.lower()
    words = re.findall(r"[a-zA-Z']+", lower)
    n_words = max(len(words), 1)

    signals = []
    points = 0.0

    # (a) Sentence-length uniformity: AI-generated text tends to have more
    # evenly-sized sentences than human writing (lower coefficient of
    # variation in sentence length).
    sentences = [s.strip() for s in re.split(r"[.!?]+", body) if s.strip()]
    sent_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences if len(re.findall(r"[a-zA-Z']+", s)) > 0]
    if len(sent_lengths) >= 3:
        mean_len = sum(sent_lengths) / len(sent_lengths)
        var = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        cv = (var ** 0.5) / mean_len if mean_len > 0 else 1.0
        a_pts = max(0.0, (0.6 - cv) / 0.6) * 25
        points += a_pts
        if a_pts > 12:
            signals.append("Sentence lengths are unusually uniform (low variability) — a pattern more typical of AI-generated text than human writing.")
    else:
        points += 12.5  # neutral contribution when too short to assess

    # (b) Stock/formulaic phrases common in LLM output
    phrase_hits = [p for p in _STOCK_PHRASES if p in lower]
    b_pts = min(len(phrase_hits), 4) / 4 * 20
    points += b_pts
    if phrase_hits:
        signals.append(f"Contains {len(phrase_hits)} common LLM-style stock phrase(s), e.g. \"{phrase_hits[0]}\".")

    # (c) Contraction usage: AI-generated formal text often avoids
    # contractions more than typical human writing.
    if n_words >= 30:
        contraction_count = sum(lower.count(c) for c in _CONTRACTIONS)
        per_100 = contraction_count / n_words * 100
        c_pts = max(0.0, (2.0 - per_100) / 2.0) * 15
        points += c_pts
        if c_pts > 10:
            signals.append("Very few or no contractions used (e.g. \"don't\", \"I'm\") — more typical of formal AI-generated text.")
    else:
        points += 7.5

    # (d) Absence of informal markers (typos/elongation, ALL-CAPS shouting,
    # repeated punctuation) that are more common in human-written emails.
    has_elongation = bool(_ELONGATION_RE.search(text))
    has_shouty = bool(_SHOUTY_RE.search(text))
    has_repeated_punct = bool(re.search(r"[!?]{2,}", text))
    if not (has_elongation or has_shouty or has_repeated_punct):
        points += 15
        signals.append("No informal markers detected (no ALL-CAPS shouting, elongated words, or repeated punctuation).")

    # (e) Lexical diversity in a "stable moderate-high" band -- literature
    # suggests AI text tends to avoid both very repetitive and very erratic
    # vocabulary use.
    diversity = len(set(words)) / n_words
    if n_words >= 20:
        if 0.55 <= diversity <= 0.85:
            points += 15
            signals.append(f"Lexical diversity ({diversity:.2f}) sits in the stable, moderate-high range often seen in AI-generated text.")
    else:
        points += 7.5

    # (f) Em-dash usage -- increasingly noted as a stylistic tell of LLM output.
    if "—" in text or " - " in text:
        points += 10
        signals.append("Uses em-dashes or dash-separated clauses, a stylistic pattern frequently observed in LLM-generated writing.")

    score = round(min(max(points, 0), 100), 1)
    return {"score": score, "signals": signals, "n_signals": len(signals)}
