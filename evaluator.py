from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from preprocess import preprocess_text
from utils import ROOT_DIR


MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def evaluate_model(data_path: Path | None = None) -> Dict[str, float]:
    """
    Load the trained model and evaluate on the full dataset.
    """
    if data_path is None:
        data_path = ROOT_DIR / "data" / "dataset.csv"

    df = pd.read_csv(data_path)
    df["Resume_Text_clean"] = df["Resume_Text"].astype(str).apply(preprocess_text)
    df["Job_Description_clean"] = df["Job_Description"].astype(str).apply(preprocess_text)
    df["combined_text"] = df["Resume_Text_clean"] + " " + df["Job_Description_clean"]

    X = df["combined_text"].values
    y_true = df["Label"].values

    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X_vec = vectorizer.transform(X)
    y_pred = clf.predict(X_vec)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate_model()

