import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


_STOPWORDS = None


def _ensure_nltk_resources() -> None:
    """
    Download required NLTK resources if they are not already available.
    This is safe to call multiple times.
    """
    global _STOPWORDS
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Basic text normalization: lowercase, remove non-alphabetic characters
    (keeping spaces), and collapse extra whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline used for both resumes and job descriptions.
    """
    _ensure_nltk_resources()
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """
    Convenience function that returns tokens after preprocessing.
    """
    processed = preprocess_text(text)
    return processed.split() if processed else []

