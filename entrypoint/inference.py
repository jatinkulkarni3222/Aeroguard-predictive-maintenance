"""
FastAPI Inference & Maintenance Copilot Service for NASA CMAPSS Predictive Maintenance.
Exposes REST endpoints for online failure prediction, sensor degradation attribution,
and RAG-assisted maintenance troubleshooting.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipelines.inference_pipeline import PredictiveMaintenanceInferenceService
from src.rag.engine import MaintenanceCopilot

# Initialize core services
inference_service = PredictiveMaintenanceInferenceService(root_dir=ROOT)
copilot = MaintenanceCopilot(kb_dir=ROOT / "src" / "rag" / "knowledge_base")

app = FastAPI(
    title="NASA Turbofan Predictive Maintenance & RAG Copilot API",
    description="Production-grade API providing real-time failure risk classification, sensor degradation attribution, and RAG-driven prescriptive maintenance advisories.",
    version="2.0.0"
)


# -------------------------
# Request / Response Schemas
# -------------------------
class FeatureInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Key-value mapping of engineered feature names to values")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "setting1": 0.0007,
                    "setting2": -0.0004,
                    "setting3": 100.0,
                    "sensor_2": 642.35,
                    "sensor_3": 1582.85,
                    "sensor_4": 1400.60,
                    "sensor_11": 47.15,
                    "sensor_3_r10_mean": 1583.10,
                    "sensor_15_r10_mean": 8.42
                }
            }
        }


class SimpleSensorInput(BaseModel):
    sensors: Dict[str, float] = Field(..., description="Mapping of base sensor readings, e.g. sensor_2, sensor_3, sensor_11")
    settings: Optional[Dict[str, float]] = Field(default=None, description="Operating settings: setting1, setting2, setting3")

    class Config:
        json_schema_extra = {
            "example": {
                "sensors": {
                    "sensor_2": 642.35,
                    "sensor_3": 1582.85,
                    "sensor_4": 1402.10,
                    "sensor_7": 553.75,
                    "sensor_8": 2388.05,
                    "sensor_9": 9050.20,
                    "sensor_11": 47.20,
                    "sensor_12": 521.66,
                    "sensor_13": 2388.02,
                    "sensor_14": 8130.40,
                    "sensor_15": 8.43,
                    "sensor_17": 392.0,
                    "sensor_20": 39.06,
                    "sensor_21": 23.41
                },
                "settings": {
                    "setting1": 0.0005,
                    "setting2": 0.0002,
                    "setting3": 100.0
                }
            }
        }


class TroubleshootRequest(BaseModel):
    unit_id: int = Field(default=1, description="Engine unit identifier")
    cycle: int = Field(default=150, description="Current operating cycle count")
    failure_risk: float = Field(default=0.85, description="Model-predicted failure probability (0.0 to 1.0)")
    risk_level: str = Field(default="CRITICAL", description="Risk level: NORMAL, WARNING, CRITICAL")
    degrading_factors: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of top anomalous sensor factors")


class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="Technical maintenance or troubleshooting question")


# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
def health_check():
    """System health check and deployment status."""
    return {
        "status": "online",
        "service": "predictive-maintenance-inference-rag",
        "model_loaded": inference_service.is_ready,
        "features_indexed": len(inference_service.feature_names),
        "version": "2.0.0"
    }


@app.get("/metrics")
def get_metrics():
    """Returns validated model evaluation metrics from offline cross-validation."""
    metrics_path = ROOT / "data" / "03-predictions" / "eval_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    raise HTTPException(status_code=404, detail="Evaluation metrics file not found. Run training first.")


@app.post("/predict_from_features")
def predict_from_features(payload: FeatureInput):
    """Predicts failure risk from a full vector of engineered features."""
    try:
        result = inference_service.predict_from_features(payload.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_simple")
def predict_simple(payload: SimpleSensorInput):
    """Predicts failure risk from raw sensor readings with smart imputation."""
    try:
        result = inference_service.predict_simple(payload.sensors, payload.settings)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rag/troubleshoot")
def troubleshoot_unit(payload: TroubleshootRequest):
    """Generates an authentic OEM Maintenance Work-Order using RAG retrieval over technical SOPs."""
    try:
        work_order = copilot.generate_work_order(
            unit_id=payload.unit_id,
            cycle=payload.cycle,
            failure_risk=payload.failure_risk,
            risk_level=payload.risk_level,
            degrading_factors=payload.degrading_factors
        )
        return work_order
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
def query_knowledge_base(payload: RAGQueryRequest):
    """Searches the technical knowledge base and synthesizes prescriptive answers for engineers."""
    try:
        response = copilot.answer_query(payload.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
