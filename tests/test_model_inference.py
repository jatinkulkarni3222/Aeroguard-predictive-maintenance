"""
MLOps Unit Tests: Model Serving, Anomaly Attribution & Prescriptive RAG.
Verifies inference contract, prediction latency bounds, and SOP retrieval.
"""

from pathlib import Path
import time
import pytest

from src.pipelines.inference_pipeline import PredictiveMaintenanceInferenceService
from src.rag.engine import MaintenanceCopilot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inference_service():
    return PredictiveMaintenanceInferenceService(root_dir=ROOT)


@pytest.fixture(scope="module")
def rag_copilot():
    return MaintenanceCopilot(kb_dir=ROOT / "src" / "rag" / "knowledge_base")


def test_inference_service_initialization(inference_service):
    """Verifies all model artifacts and feature contracts load correctly."""
    assert inference_service.is_ready
    assert len(inference_service.feature_names) == 297
    assert inference_service.model is not None


def test_simple_prediction_contract(inference_service):
    """Verifies online single-reading inference and probability bounds."""
    telemetry = {
        "sensor_2": 642.35, "sensor_3": 1582.85, "sensor_4": 1402.10,
        "sensor_7": 553.75, "sensor_8": 2388.05, "sensor_9": 9050.20,
        "sensor_11": 47.20, "sensor_12": 521.66, "sensor_13": 2388.02,
        "sensor_14": 8130.40, "sensor_15": 8.43, "sensor_17": 392.0,
        "sensor_20": 39.06, "sensor_21": 23.41
    }
    settings = {"setting1": 0.0005, "setting2": 0.0002, "setting3": 100.0}

    t0 = time.time()
    result = inference_service.predict_simple(telemetry, settings)
    latency_ms = (time.time() - t0) * 1000.0

    # Latency requirement: sub-150ms inference
    assert latency_ms < 150.0, f"Inference too slow: {latency_ms:.2f} ms"
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["failure_risk"] <= 1.0
    assert result["risk_level"] in ["NORMAL", "WARNING", "CRITICAL"]
    assert "top_degrading_factors" in result


def test_rag_copilot_work_order_generation(rag_copilot):
    """Verifies that RAG pipeline retrieves relevant SOPs and produces structured work orders."""
    anomalies = [
        {"sensor": "sensor_11", "deviation_direction": "DEPRESSED", "severity": "HIGH"},
        {"sensor": "sensor_4", "deviation_direction": "ELEVATED", "severity": "HIGH"}
    ]
    wo = rag_copilot.generate_work_order(
        unit_id=12,
        cycle=175,
        failure_risk=0.88,
        risk_level="CRITICAL",
        degrading_factors=anomalies
    )

    assert "work_order_id" in wo
    assert "P1 - IMMEDIATE_AOG" in wo["dispatch_priority"]
    assert len(wo["matched_sops"]) > 0
    assert len(wo["mandatory_action_checklist"]) > 0
    assert len(wo["recommended_parts"]) > 0
