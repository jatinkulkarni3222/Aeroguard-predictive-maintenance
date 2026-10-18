"""
Model Training & Evaluation Pipeline for Predictive Maintenance.
Trains a balanced Random Forest model on CMAPSS multivariate sensor telemetry,
using GroupShuffleSplit by engine unit to strictly avoid temporal and unit leakage.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.pipelines.feature_pipeline import generate_features


def load_or_generate_features(root: Path) -> pd.DataFrame:
    """Loads features CSV or generates it from preprocessed data if absent."""
    feat_path = root / "data" / "03-features" / "FD001_features.csv"
    prep_path = root / "data" / "02-preprocessed" / "FD001_preprocessed.csv"

    if feat_path.exists():
        print(f"Loading cached features: {feat_path}")
        return pd.read_csv(feat_path)
    
    if not prep_path.exists():
        raise FileNotFoundError(f"Neither {feat_path} nor {prep_path} exists.")
    
    print(f"Generating features from {prep_path}...")
    df_raw = pd.read_csv(prep_path)
    df_feat = generate_features(df_raw)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(feat_path, index=False)
    return df_feat


def run_training_pipeline(
    root_dir: Path = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 150,
    max_depth: int = 12
) -> Dict[str, Any]:
    """
    Executes full data preparation, group-split, model training, artifact persistence,
    and comprehensive evaluation.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[2]

    out_model_dir = root_dir / "models"
    out_pred_dir = root_dir / "data" / "03-predictions"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_features(root_dir)

    # Validate target presence
    target_col = "failure_within_30"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    drop_cols = {"unit", "time_cycle", "max_cycle", "RUL", target_col}
    feature_names = [c for c in df.columns if c not in drop_cols]

    X = df[feature_names].copy()
    y = df[target_col].values.astype(int)
    groups = df["unit"].values

    print(f"Dataset summary: {len(df)} cycles across {len(np.unique(groups))} engines.")
    print(f"Feature count: {len(feature_names)} features.")
    print(f"Class distribution: Healthy (0) = {(y == 0).sum()}, Failing (1) = {(y == 1).sum()} ({((y == 1).sum() / len(y) * 100):.1f}% positive)")

    # 1. Group Shuffle Split (train units vs holdout validation units)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    train_units = np.unique(groups[train_idx])
    val_units = np.unique(groups[val_idx])
    print(f"Split: {len(train_units)} engines for training, {len(val_units)} engines for evaluation.")

    X_train_raw, y_train = X.iloc[train_idx], y[train_idx]
    X_val_raw, y_val = X.iloc[val_idx], y[val_idx]

    # 2. Imputation and Scaling (fitted strictly on training set)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_names)
    X_train = scaler.fit_transform(X_train_imp)

    X_val_imp = pd.DataFrame(imputer.transform(X_val_raw), columns=feature_names)
    X_val = scaler.transform(X_val_imp)

    # Persist preprocessors
    joblib.dump(imputer, out_model_dir / "imputer_joblib.pkl")
    joblib.dump(scaler, out_model_dir / "scaler_joblib.pkl")

    # 3. Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Persist model & feature names
    joblib.dump(clf, out_model_dir / "rf_failure_joblib.pkl")
    (out_model_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    # 4. Comprehensive Evaluation
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_val, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_val, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "sample_counts": {
            "total_cycles_evaluated": int(len(y_val)),
            "validation_units_count": int(len(val_units)),
            "training_units_count": int(len(train_units))
        }
    }

    # Save metrics JSON
    (out_pred_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save predictions dataframe
    val_df = df.iloc[val_idx].copy()
    val_report = pd.DataFrame({
        "unit": val_df["unit"].values,
        "time_cycle": val_df["time_cycle"].values,
        "RUL": val_df["RUL"].values,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_score": np.round(y_prob, 4)
    })
    val_report.to_csv(out_pred_dir / "eval_report.csv", index=False)

    # Save feature importances
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(out_pred_dir / "feature_importances.csv", index=False)

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
        "features_count": len(feature_names),
        "classes": [int(c) for c in clf.classes_],
        "metrics": metrics,
        "train_units": [int(u) for u in train_units],
        "validation_units": [int(u) for u in val_units]
    }
    (out_model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Training and evaluation completed successfully!")
    print("Summary Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nTop 10 Important Features:")
    print(fi_df.head(10).to_string(index=False))

    return {
        "model": clf,
        "imputer": imputer,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importances": fi_df
    }


if __name__ == "__main__":
    run_training_pipeline()
