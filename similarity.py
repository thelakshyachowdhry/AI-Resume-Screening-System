from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with sensible defaults for resume text.
    """
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )


def compute_job_resume_similarity(
    job_description: str,
    resumes: List[str],
) -> List[float]:
    """
    Compute cosine similarity between a single job description and many resumes.
    """
    if not job_description or not resumes:
        return [0.0 for _ in resumes]

    docs = [preprocess_text(job_description)] + [preprocess_text(r) for r in resumes]
    vectorizer = build_tfidf_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    job_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(job_vec, resume_vecs)[0]
    return [round(float(s), 4) for s in sims]


def compute_resume_to_resume_similarity(resumes: List[str]) -> List[Tuple[int, int, float]]:
    """
    Pairwise similarity between resumes for duplicate detection.

    Returns a list of tuples: (idx_i, idx_j, similarity).
    """
    n = len(resumes)
    if n < 2:
        return []

    processed = [preprocess_text(r) for r in resumes]
    vectorizer = build_tfidf_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed)

    sim_matrix = cosine_similarity(tfidf_matrix)
    pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, round(float(sim_matrix[i, j]), 4)))
    return pairs


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    """
    Lazily load and cache SentenceTransformer model.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_bert_similarity(
    job_description: str,
    resumes: List[str],
) -> Tuple[List[float], str | None]:
    """
    Compute semantic similarity (BERT embeddings) between job and resumes.

    Returns
    -------
    Tuple[List[float], str | None]
        - List of cosine similarity scores in [0, 1].
        - Error message if model failed, else None.
    """
    if not job_description or not resumes:
        return [0.0 for _ in resumes], None

    try:
        model = _load_sentence_transformer()
        docs = [job_description] + resumes
        embeddings = model.encode(
            docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        job_vec = embeddings[0]
        resume_vecs = embeddings[1:]

        # With normalized vectors, cosine similarity equals dot product.
        sims = np.dot(resume_vecs, job_vec)
        sims = np.clip(sims, 0.0, 1.0)
        return [round(float(s), 4) for s in sims], None
    except Exception as e:
        return [0.0 for _ in resumes], str(e)

