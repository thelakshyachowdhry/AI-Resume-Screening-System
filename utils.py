from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_DIR / "logs.txt"


def setup_logging() -> None:
    """
    Configure a simple file + console logger for the project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def log_event(event: str, extras: Optional[Dict[str, Any]] = None) -> None:
    """
    Append a structured log line to the log file.
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "extras": extras or {},
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def safe_load_joblib(path: Path) -> Any:
    """
    Load a joblib file if it exists; return None otherwise.
    """
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@dataclass
class ScoreBreakdown:
    similarity_score: float  # 0–100
    skills_score: float  # 0–100
    model_probability: Optional[float]  # 0–1, may be None
    final_score: float  # 0–100


def compute_final_score(
    similarity_score: float,
    skills_score: float,
    model_probability: Optional[float] = None,
    w_similarity: float = 0.4,
    w_skills: float = 0.4,
    w_model: float = 0.2,
) -> ScoreBreakdown:
    """
    Combine different signals into a single final score.
    """
    sim_pct = max(0.0, min(1.0, similarity_score))
    skills_pct = max(0.0, min(100.0, skills_score)) / 100.0

    if model_probability is None:
        w_similarity = w_similarity + w_model / 2
        w_skills = w_skills + w_model / 2
        w_model = 0.0
        model_probability = 0.0

    total = w_similarity + w_skills + w_model
    if total == 0:
        total = 1.0

    final = (
        (w_similarity / total) * sim_pct
        + (w_skills / total) * skills_pct
        + (w_model / total) * float(model_probability)
    )

    return ScoreBreakdown(
        similarity_score=round(sim_pct * 100, 2),
        skills_score=round(skills_pct * 100, 2),
        model_probability=round(float(model_probability), 4)
        if model_probability is not None
        else None,
        final_score=round(final * 100, 2),
    )


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for Streamlit download buttons.
    """
    return df.to_csv(index=False).encode("utf-8")
