#!/usr/bin/env python3
"""
Entrypoint script for model training and offline validation.
Runs the complete feature extraction, group-split cross validation,
and saves production artifacts to models/ and data/03-predictions/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.training_pipeline import run_training_pipeline


def main():
    print("=" * 60)
    print("  NASA CMAPSS PREDICTIVE MAINTENANCE: TRAINING PIPELINE")
    print("=" * 60)
    
    results = run_training_pipeline(
        root_dir=ROOT,
        test_size=0.2,
        random_state=42,
        n_estimators=150,
        max_depth=12
    )

    print("\n[SUCCESS] Artifacts persisted:")
    print(f" - Model: {ROOT / 'models' / 'rf_failure_joblib.pkl'}")
    print(f" - Metrics: {ROOT / 'data' / '03-predictions' / 'eval_metrics.json'}")
    print(f" - Feature Importances: {ROOT / 'data' / '03-predictions' / 'feature_importances.csv'}")
    print(f" - Evaluation Report: {ROOT / 'data' / '03-predictions' / 'eval_report.csv'}")


if __name__ == "__main__":
    main()
