"""
Student pass/fail prediction — decision tree (entropy, max depth 5).

Follows Joshua Eckroth, Packt Ch. 1 (Student performance.py): UCI student-por data,
pass if G1+G2+G3 >= 35, one-hot categoricals, 500-row train split, sklearn tree.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split

MAX_DEPTH = 5
TRAIN_HEAD = 500
RANDOM_SEED = 42
PASS_GRADE_SUM = 35

CATEGORICAL_COLUMNS = (
    "sex",
    "school",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
)

NUMERIC_COLUMNS = (
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
)

INPUT_FIELDS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def default_data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "student-por.csv"


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "web" / "data" / "cache"


def load_raw_frame(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or default_data_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path, sep=";")


def prepare_labeled_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["pass"] = frame.apply(
        lambda row: 1 if (row["G1"] + row["G2"] + row["G3"]) >= PASS_GRADE_SUM else 0,
        axis=1,
    )
    frame = frame.drop(columns=["G1", "G2", "G3"])
    frame = pd.get_dummies(
        frame,
        columns=list(CATEGORICAL_COLUMNS),
        dtype=int,
    )
    return frame.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def encode_student_row(row: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """One student (raw fields) → aligned feature matrix."""
    base = {col: row.get(col) for col in INPUT_FIELDS}
    for col in NUMERIC_COLUMNS:
        base[col] = int(base.get(col, 0))
    for col in CATEGORICAL_COLUMNS:
        base[col] = str(base.get(col, ""))

    single = pd.DataFrame([base])
    encoded = pd.get_dummies(single, columns=list(CATEGORICAL_COLUMNS), dtype=int)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_columns]
    return encoded


@dataclass
class StudentModel:
    classifier: tree.DecisionTreeClassifier
    feature_columns: list[str]
    holdout_accuracy: float
    cv_mean: float
    cv_std: float
    pass_rate: float
    sample_students: list[dict[str, Any]]

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        features = encode_student_row(row, self.feature_columns)
        proba = self.classifier.predict_proba(features)[0]
        label = int(self.classifier.predict(features)[0])
        classes = list(self.classifier.classes_)
        fail_idx = classes.index(0) if 0 in classes else 0
        pass_idx = classes.index(1) if 1 in classes else 1
        return {
            "prediction": "pass" if label == 1 else "fail",
            "probabilities": {
                "fail": round(100 * float(proba[fail_idx]), 1),
                "pass": round(100 * float(proba[pass_idx]), 1),
            },
        }


def train_model(data_path: Path | None = None) -> StudentModel:
    raw = load_raw_frame(data_path)
    prepared = prepare_labeled_frame(raw)
    features = prepared.drop(columns=["pass"])
    labels = prepared["pass"]

    d_train = prepared.iloc[:TRAIN_HEAD]
    d_test = prepared.iloc[TRAIN_HEAD:]
    x_train = d_train.drop(columns=["pass"])
    y_train = d_train["pass"]
    x_test = d_test.drop(columns=["pass"])
    y_test = d_test["pass"]

    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=MAX_DEPTH)
    clf.fit(x_train, y_train)
    holdout = float(clf.score(x_test, y_test))

    scores = cross_val_score(clf, features, labels, cv=5)
    cv_mean = float(scores.mean())
    cv_std = float(scores.std() * 2)

    pass_rate = float(labels.sum()) / len(labels)

    samples: list[dict[str, Any]] = []
    raw_shuffled = raw.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    for idx in random.Random(RANDOM_SEED).sample(range(len(raw_shuffled)), k=min(6, len(raw_shuffled))):
        row = raw_shuffled.iloc[idx]
        raw_fields = {col: row[col] for col in INPUT_FIELDS}
        actual_pass = 1 if (row["G1"] + row["G2"] + row["G3"]) >= PASS_GRADE_SUM else 0
        pred = clf.predict(encode_student_row(raw_fields, list(x_train.columns)))[0]
        samples.append(
            {
                "label": f"Student {idx + 1}",
                "fields": _serialize_fields(raw_fields),
                "actual": "pass" if actual_pass == 1 else "fail",
                "predicted": "pass" if pred == 1 else "fail",
            }
        )

    return StudentModel(
        classifier=clf,
        feature_columns=list(x_train.columns),
        holdout_accuracy=round(100 * holdout, 1),
        cv_mean=round(100 * cv_mean, 1),
        cv_std=round(100 * cv_std, 1),
        pass_rate=round(100 * pass_rate, 1),
        sample_students=samples,
    )


def _serialize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = int(value) if float(value).is_integer() else float(value)
        else:
            out[key] = value
    return out


def save_model(model: StudentModel, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "model.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    meta = {
        "holdout_accuracy": model.holdout_accuracy,
        "cv_mean": model.cv_mean,
        "cv_std": model.cv_std,
        "pass_rate": model.pass_rate,
        "max_depth": MAX_DEPTH,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_cached_model(cache_dir: Path) -> StudentModel | None:
    path = cache_dir / "model.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def get_or_train_model(
    data_path: Path | None = None,
    cache_dir: Path | None = None,
) -> StudentModel:
    cache = cache_dir or default_cache_dir()
    cached = load_cached_model(cache)
    if cached is not None:
        return cached
    model = train_model(data_path)
    save_model(model, cache)
    return model


def field_options(raw: pd.DataFrame | None = None) -> dict[str, list[Any]]:
    frame = raw if raw is not None else load_raw_frame()
    options: dict[str, list[Any]] = {}
    for col in NUMERIC_COLUMNS:
        options[col] = sorted(int(v) for v in frame[col].unique())
    for col in CATEGORICAL_COLUMNS:
        options[col] = sorted(frame[col].astype(str).unique().tolist())
    return options


def default_student_fields(options: dict[str, list[Any]]) -> dict[str, Any]:
    """Sensible defaults for the form."""
    return {
        "age": 16,
        "Medu": 2,
        "Fedu": 2,
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 2,
        "sex": options["sex"][0] if options.get("sex") else "F",
        "school": "GP",
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
    }
