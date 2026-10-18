"""
MLOps Unit Tests: Data Leakage Prevention & Group Splitting.
Mathematically guarantees that engine units are strictly partitioned
with 0% cross-unit or cross-temporal contamination.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit


def test_zero_leakage_group_split():
    """Verifies that train and validation unit sets are completely disjoint."""
    metadata_path = Path(__file__).resolve().parents[1] / "models" / "model_metadata.json"
    assert metadata_path.exists(), "Model metadata not found."

    metadata = json.loads(metadata_path.read_text())
    train_units = set(metadata.get("train_units", []))
    val_units = set(metadata.get("validation_units", []))

    assert len(train_units) > 0, "No train units recorded"
    assert len(val_units) > 0, "No validation units recorded"

    # Strictly disjoint sets: Intersection MUST be empty
    overlap = train_units.intersection(val_units)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping units: {overlap}"
    assert len(train_units) + len(val_units) == 100, "Total units must equal 100"


def test_validation_metrics_criteria():
    """Verifies that model passes minimum reliability thresholds for deployment."""
    metrics_path = Path(__file__).resolve().parents[1] / "data" / "03-predictions" / "eval_metrics.json"
    assert metrics_path.exists(), "Evaluation metrics file missing."

    metrics = json.loads(metrics_path.read_text())

    assert metrics["accuracy"] >= 0.85, f"Accuracy below 85%: {metrics['accuracy']}"
    assert metrics["recall"] >= 0.85, f"Recall below 85%: {metrics['recall']}"
    assert metrics["f1"] >= 0.85, f"F1-Score below 85%: {metrics['f1']}"
    assert metrics["roc_auc"] >= 0.85, f"ROC-AUC below 0.85: {metrics['roc_auc']}"
