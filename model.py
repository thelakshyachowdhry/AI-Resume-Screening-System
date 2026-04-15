from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import preprocess_text
from utils import ROOT_DIR, setup_logging, log_event


MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"Resume_Text", "Job_Description", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Resume_Text_clean"] = df["Resume_Text"].astype(str).apply(preprocess_text)
    df["Job_Description_clean"] = df["Job_Description"].astype(str).apply(preprocess_text)
    df["combined_text"] = df["Resume_Text_clean"] + " " + df["Job_Description_clean"]
    return df


def train_model(data_path: Path | None = None) -> None:
    """
    Train a Logistic Regression classifier on resume–job pairs.
    """
    setup_logging()
    if data_path is None:
        data_path = ROOT_DIR / "data" / "dataset.csv"

    log_event("training_started", {"dataset": str(data_path)})
    df = load_dataset(data_path)
    df = preprocess_dataset(df)

    X = df["combined_text"].values
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    log_event("training_finished", {"accuracy": float(acc)})
    print(f"Accuracy: {acc:.4f}")

    # Persist model and vectorizer separately for transparency in the project.
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()

