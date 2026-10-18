"""
Inference Pipeline Service for NASA CMAPSS Predictive Maintenance.
Handles single-sample prediction, batch unit evaluation, and sensor anomaly attribution.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import joblib
import numpy as np
import pandas as pd


class PredictiveMaintenanceInferenceService:
    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[2]
        self.root_dir = root_dir
        self.model_dir = root_dir / "models"
        self.data_dir = root_dir / "data"

        # Artifact paths
        self.model_path = self.model_dir / "rf_failure_joblib.pkl"
        self.imputer_path = self.model_dir / "imputer_joblib.pkl"
        self.scaler_path = self.model_dir / "scaler_joblib.pkl"
        self.feature_names_path = self.model_dir / "feature_names.json"
        self.metadata_path = self.model_dir / "model_metadata.json"

        # Lazy loaded components
        self.model = None
        self.imputer = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Please execute `python entrypoint/train.py` first."
            )
        self.model = joblib.load(self.model_path)
        self.imputer = joblib.load(self.imputer_path) if self.imputer_path.exists() else None
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None

        if self.feature_names_path.exists():
            self.feature_names = json.loads(self.feature_names_path.read_text())
        else:
            # Fallback to feature importances csv
            fi_path = self.data_dir / "03-predictions" / "feature_importances.csv"
            if fi_path.exists():
                self.feature_names = list(pd.read_csv(fi_path)["feature"].values)

        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text())

    @property
    def is_ready(self) -> bool:
        return self.model is not None and len(self.feature_names) > 0

    def _determine_risk_level(self, probability: float) -> str:
        if probability >= 0.70:
            return "CRITICAL"
        elif probability >= 0.35:
            return "WARNING"
        return "NORMAL"

    def predict_from_features(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs inference on a complete dictionary of engineered features.
        """
        if not self.is_ready:
            raise RuntimeError("Inference service is not initialized.")

        # Ensure correct ordering and handle missing features with median from imputer
        row = []
        for name in self.feature_names:
            if name in features_dict:
                row.append(float(features_dict[name]))
            else:
                row.append(np.nan)

        X_df = pd.DataFrame([row], columns=self.feature_names)

        if self.imputer is not None:
            try:
                X_df = pd.DataFrame(self.imputer.transform(X_df), columns=self.feature_names)
            except Exception:
                X_df = X_df.fillna(0.0)
        else:
            X_df = X_df.fillna(0.0)

        if self.scaler is not None:
            try:
                X_arr = self.scaler.transform(X_df)
            except Exception:
                X_arr = X_df.values
        else:
            X_arr = X_df.values

        pred_class = int(self.model.predict(X_arr)[0])
        probabilities = self.model.predict_proba(X_arr)[0]
        failure_risk = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        risk_level = self._determine_risk_level(failure_risk)

        # Identify anomalous/degraded sensors based on standardized feature values
        deviations = self._find_top_deviations(X_arr[0], top_k=5)

        return {
            "prediction": pred_class,
            "failure_risk": round(failure_risk, 4),
            "risk_level": risk_level,
            "estimated_cycles_to_failure": int(max(0, round((1.0 - failure_risk) * 60 + 5))) if failure_risk > 0.5 else ">30",
            "top_degrading_factors": deviations
        }

    def predict_simple(self, sensors: Dict[str, float], settings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Runs inference from raw base sensor readings. Imputes rolling statistics
        using current sensor values and nominal defaults.
        """
        features: Dict[str, float] = {}

        # Add settings
        if settings:
            for k, v in settings.items():
                features[k] = float(v)

        # Add base sensors
        for k, v in sensors.items():
            val = float(v)
            features[k] = val
            # Approximate rolling features with current snapshot
            for w in [5, 10, 20]:
                features[f"{k}_r{w}_mean"] = val
                features[f"{k}_r{w}_std"] = 0.0
                features[f"{k}_r{w}_min"] = val
                features[f"{k}_r{w}_max"] = val
            features[f"{k}_trend"] = 0.0

        return self.predict_from_features(features)

    def _find_top_deviations(self, scaled_features: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Computes top contributing abnormal sensor indicators using absolute Z-score deviation.
        """
        if len(scaled_features) != len(self.feature_names):
            return []

        abs_devs = np.abs(scaled_features)
        sorted_indices = np.argsort(abs_devs)[::-1]

        factors = []
        seen_sensors = set()

        for idx in sorted_indices:
            feat_name = self.feature_names[idx]
            z_score = float(scaled_features[idx])
            
            # Extract root sensor name
            sensor_tag = feat_name.split("_r")[0].split("_trend")[0]
            if sensor_tag not in seen_sensors:
                seen_sensors.add(sensor_tag)
                factors.append({
                    "feature": feat_name,
                    "sensor": sensor_tag,
                    "z_score": round(z_score, 2),
                    "deviation_direction": "ELEVATED" if z_score > 0 else "DEPRESSED",
                    "severity": "HIGH" if abs(z_score) > 2.0 else "MODERATE"
                })
            if len(factors) >= top_k:
                break

        return factors

    def load_unit_history(self, unit_id: int, df_source: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Loads feature history for a specific unit and computes cycle-by-cycle risk trajectory.
        Accepts optional df_source to avoid reading from disk.
        """
        if df_source is not None:
            df = df_source
        else:
            feat_path = self.data_dir / "03-features" / "FD001_features.csv"
            if not feat_path.exists():
                return None
            df = pd.read_csv(feat_path)

        unit_df = df[df["unit"] == unit_id].sort_values("time_cycle").copy()
        if unit_df.empty:
            return None

        drop_cols = {"unit", "time_cycle", "max_cycle", "RUL", "failure_within_30"}
        feat_cols = [c for c in unit_df.columns if c not in drop_cols]

        X_unit = unit_df[feat_cols]
        if self.imputer is not None:
            try:
                X_unit = pd.DataFrame(self.imputer.transform(X_unit), columns=feat_cols)
            except Exception:
                X_unit = X_unit.fillna(0.0)
        else:
            X_unit = X_unit.fillna(0.0)

        if self.scaler is not None:
            try:
                X_arr = self.scaler.transform(X_unit)
            except Exception:
                X_arr = X_unit.values
        else:
            X_arr = X_unit.values

        probs = self.model.predict_proba(X_arr)[:, 1]
        preds = self.model.predict(X_arr)

        unit_df["predicted_risk"] = np.round(probs, 4)
        unit_df["predicted_label"] = preds
        unit_df["risk_level"] = [self._determine_risk_level(p) for p in probs]

        return unit_df
