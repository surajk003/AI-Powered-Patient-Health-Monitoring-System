

from pathlib import Path
import argparse
import json
import sys
import os
import time
import logging
import traceback
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df


def build_preprocessor(X: pd.DataFrame):
    # automatic column selectors
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # treat bools and object-like as categorical
    categorical_cols = X.select_dtypes(include=[object, "category", bool]).columns.tolist()

    # remove any columns that may be target-like from features selection (defensive)
    # caller ensures target removed before sending X

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Build OneHotEncoder with the correct sparse/dense keyword depending on scikit-learn version
        (
            "onehot",
            OneHotEncoder(**({"handle_unknown": "ignore", "sparse": False} if "sparse" in inspect.signature(OneHotEncoder).parameters else {"handle_unknown": "ignore", "sparse_output": False})
            ),
        ),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names_from_column_transformer(column_transformer, numeric_cols, categorical_cols):
    # Build feature names list after ColumnTransformer transformation
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            # OneHotEncoder case
            ohe = transformer.named_steps["onehot"]
            try:
                names = list(ohe.get_feature_names_out(cols))
            except Exception:
                # fallback
                names = []
                for c in cols:
                    # we don't know categories easily here; use column name placeholder
                    names.append(c)
            feature_names.extend(names)
        else:
            # numeric or other transformer producing same number of columns
            feature_names.extend(cols)
    return feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default=str(Path(__file__).resolve().parent / "Synthetic_patient-HealthCare-Monitoring_dataset.csv"),
                        help="Path to the CSV dataset")
    parser.add_argument("--target", type=str, default="Predicted Disease", help="Name of the target column")
    parser.add_argument("--output", type=str, default="patient_monitoring_rf_model.pkl", help="Filename for saved model")
    args = parser.parse_args()

    data_path = Path(args.data_path)

    # --- Logging setup -------------------------------------------------
    log_file = Path("train_run.log")
    # Create logger that writes to both console and a run-specific log file
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    # Avoid adding duplicate handlers if main is called multiple times
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.info(f"Starting training run. Data path: {data_path}")

    try:
        df = load_data(data_path)
        logger.info(f"Data shape: {df.shape}")
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        raise

    if args.target not in df.columns:
        logger.error("target column '%s' not found in dataset columns: %s", args.target, df.columns.tolist())
        sys.exit(1)

    # Basic cleaning: drop rows where target is missing
    df = df.dropna(subset=[args.target])

    # Separate X and y
    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    class_names = le.classes_.tolist()
    logger.info("Detected classes (%d): %s", len(class_names), class_names)

    # Build preprocessor automatically
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    logger.info("Numeric cols (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical cols (%d): %s", len(categorical_cols), categorical_cols)

    # Create full pipeline with classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    # Train/test split
    stratify_arg = y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=stratify_arg)
    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    # Fit
    logger.info("Training RandomForestClassifier...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    t1 = time.time()
    logger.info("Training complete (%.2f s).", t1 - t0)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy: %.4f", acc)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = Path("confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info("Saved confusion matrix to %s", cm_path)

    # Feature importances
    # Extract feature names after preprocessing to align with importances
    try:
        ct = pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names_from_column_transformer(ct, numeric_cols, categorical_cols)
    except Exception as e:
        print(f"Warning: could not extract feature names from preprocessor: {e}")
        feature_names = None

    # Get feature importances from trained RandomForest
    try:
        rf = pipeline.named_steps["clf"]
        importances = rf.feature_importances_
        if feature_names is None or len(feature_names) != len(importances):
            # fallback: create generic names
            feature_names = [f"f_{i}" for i in range(len(importances))]

        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, max(4, 0.3 * len(fi_df))))
        sns.barplot(x="importance", y="feature", data=fi_df.head(30))
        plt.title("Top feature importances")
        plt.tight_layout()
        fi_path = Path("feature_importances.png")
        plt.savefig(fi_path)
        logger.info("Saved feature importances to %s", fi_path)
    except Exception as e:
        logger.exception("Warning: could not compute or plot feature importances: %s", e)

    # Save the pipeline and label encoder together
    out = {
        "model_pipeline": pipeline,
        "label_encoder": le,
        "feature_names": feature_names,
        "class_names": class_names
    }
    try:
        joblib.dump(out, args.output)
        logger.info("Saved trained model and artifacts to: %s", args.output)
    except Exception as e:
        logger.exception("Failed to save model: %s", e)
        raise

    logger.info("Run completed successfully.")


if __name__ == "__main__":
    main()
