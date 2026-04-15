from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from preprocess import preprocess_text


# A compact but reasonably rich skill dictionary.
SKILL_DICTIONARY: Dict[str, List[str]] = {
    "programming_languages": [
        "python",
        "java",
        "c++",
        "c",
        "javascript",
        "typescript",
        "r",
        "sql",
        "scala",
    ],
    "ml_and_ai": [
        "machine learning",
        "deep learning",
        "neural networks",
        "classification",
        "regression",
        "clustering",
        "nlp",
        "natural language processing",
        "computer vision",
        "time series",
    ],
    "ml_libraries": [
        "scikit-learn",
        "sklearn",
        "tensorflow",
        "keras",
        "pytorch",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ],
    "data_engineering": [
        "sql",
        "etl",
        "data pipeline",
        "data warehouse",
        "spark",
        "hadoop",
        "airflow",
    ],
    "cloud_and_devops": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "ci/cd",
        "git",
        "linux",
    ],
    "soft_skills": [
        "communication",
        "teamwork",
        "leadership",
        "problem solving",
        "critical thinking",
        "presentation",
    ],
}


@dataclass
class SkillAnalysis:
    detected_skills: Set[str]
    required_skills: Set[str]
    missing_skills: Set[str]
    match_score: float  # 0–100 percentage


def _normalize_for_matching(text: str) -> str:
    # Use the same preprocessing pipeline to align with similarity model
    return preprocess_text(text)


def _flatten_skill_dict() -> Set[str]:
    skills: Set[str] = set()
    for skill_list in SKILL_DICTIONARY.values():
        for skill in skill_list:
            skills.add(skill.lower())
    return skills


ALL_KNOWN_SKILLS: Set[str] = _flatten_skill_dict()


def extract_skills_from_text(text: str) -> Set[str]:
    """
    Simple keyword-based skill extractor.
    """
    if not text:
        return set()

    normalized = _normalize_for_matching(text)
    detected: Set[str] = set()

    # Match by substring for multi-word skills and simple tokens.
    for skill in ALL_KNOWN_SKILLS:
        if skill in normalized:
            detected.add(skill)

    return detected


def extract_required_skills_from_job_description(job_description: str) -> Set[str]:
    """
    Use the same heuristic extractor for required skills.
    """
    return extract_skills_from_text(job_description)


def analyze_skill_match(
    resume_text: str,
    job_description: str,
) -> SkillAnalysis:
    """
    Compute detected skills, required skills, gaps, and percentage match.
    """
    detected = extract_skills_from_text(resume_text)
    required = extract_required_skills_from_job_description(job_description)

    if not required:
        # Avoid divide-by-zero; if no skills are requested, treat as full match.
        match_score = 100.0 if detected else 0.0
        missing: Set[str] = set()
    else:
        missing = required - detected
        covered = required & detected
        match_score = (len(covered) / len(required)) * 100.0

    return SkillAnalysis(
        detected_skills=detected,
        required_skills=required,
        missing_skills=missing,
        match_score=round(match_score, 2),
    )


def skill_sets_to_strings(skills: Set[str]) -> List[str]:
    """
    Utility to format skill sets nicely for display.
    """
    return sorted(skills)

