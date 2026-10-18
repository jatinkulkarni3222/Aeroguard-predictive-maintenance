"""
RAG Maintenance Copilot & Prescriptive Work-Order Generator.
Bridges ML failure probabilities and sensor anomalies with authentic OEM
Standard Operating Procedures (SOP) and Root Cause Analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime
from src.rag.retriever import TechnicalManualRetriever


class MaintenanceCopilot:
    def __init__(self, kb_dir: Optional[Path] = None):
        self.retriever = TechnicalManualRetriever(kb_dir=kb_dir)

    def generate_work_order(
        self,
        unit_id: int,
        cycle: int,
        failure_risk: float,
        risk_level: str,
        degrading_factors: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes ML telemetry predictions and retrieved OEM documents
        into an actionable, industry-grade Maintenance Work Order.
        """
        degrading_factors = degrading_factors or []
        sensors_list = [f.get("sensor", "") for f in degrading_factors if f.get("sensor")]
        
        # Build contextual query from sensor anomalies
        if sensors_list:
            query = f"{' '.join(sensors_list)} degradation failure overhaul inspection"
        else:
            query = "compressor turbine sensor failure degradation"

        # Retrieve relevant maintenance manuals
        retrieved = self.retriever.retrieve(query, top_k=2)

        # If risk is critical, also retrieve emergency shutdown protocol
        if risk_level == "CRITICAL":
            emerg_doc = self.retriever.retrieve("critical emergency overhaul AOG ground abort", top_k=1)
            if emerg_doc and emerg_doc[0]["document"]["doc_id"] not in [r["document"]["doc_id"] for r in retrieved]:
                retrieved.append(emerg_doc[0])

        # Priority mapping
        priority_map = {
            "CRITICAL": "P1 - IMMEDIATE_AOG (AIRCRAFT ON GROUND)",
            "WARNING": "P2 - SCHEDULED_LINE_MAINTENANCE (< 25 CYCLES)",
            "NORMAL": "P3 - ROUTINE_CYCLE_MONITORING"
        }
        priority = priority_map.get(risk_level, "P3 - ROUTINE_CYCLE_MONITORING")

        # Compile SOPs and checklist
        matched_sops = []
        action_checklist = []
        parts_list = []
        root_causes = []

        for item in retrieved:
            doc = item["document"]
            matched_sops.append({
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "subsystem": doc.get("subsystem"),
                "relevance_score": item.get("score")
            })
            root_causes.extend(doc.get("root_causes", []))
            for step in doc.get("inspection_procedure", []):
                if step not in action_checklist:
                    action_checklist.append(step)
            for part in doc.get("replacement_parts", []):
                if part not in parts_list:
                    parts_list.append(part)

        work_order_id = f"WO-CMAPSS-U{unit_id:03d}-C{cycle:03d}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # Create human-readable summary
        if degrading_factors:
            sensor_desc = ", ".join([f"{f['sensor']} ({f['deviation_direction']})" for f in degrading_factors[:3]])
        else:
            sensor_desc = "Standard cycle baseline"

        telemetry_summary = (
            f"Engine Unit #{unit_id} evaluated at Cycle {cycle}. "
            f"ML Model predicts {failure_risk * 100:.1f}% risk of failure within 30 flight cycles. "
            f"Primary degraded telemetry indicators: {sensor_desc}."
        )

        return {
            "work_order_id": work_order_id,
            "timestamp": now_str,
            "unit_id": unit_id,
            "operating_cycle": cycle,
            "failure_risk_probability": round(failure_risk, 4),
            "risk_level": risk_level,
            "dispatch_priority": priority,
            "telemetry_summary": telemetry_summary,
            "identified_root_causes": list(dict.fromkeys(root_causes))[:4],
            "matched_sops": matched_sops,
            "mandatory_action_checklist": action_checklist[:6],
            "recommended_parts": parts_list[:5],
            "safety_protocol": retrieved[0]["document"].get("safety_advisories", "Follow standard FAA/EASA Part 145 PPE protocols.") if retrieved else "Follow standard protocols."
        }

    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Conversational troubleshooting assistant: retrieves matching maintenance documentation
        and synthesizes an authoritative technical response.
        """
        results = self.retriever.retrieve(question, top_k=2)
        if not results:
            return {
                "answer": "No specific maintenance procedure found for this query in the current technical knowledge base.",
                "citations": []
            }

        top_doc = results[0]["document"]
        citations = [r["document"]["doc_id"] + ": " + r["document"]["title"] for r in results]

        # Generate synthesized technical answer
        answer = (
            f"**Diagnostic Summary ({top_doc.get('doc_id')})**:\n"
            f"- **Subsystem**: {top_doc.get('subsystem')}\n"
            f"- **Key Telemetry Triggers**: {', '.join([f'{k}: {v}' for k, v in top_doc.get('telemetry_indicators', {}).items()])}\n"
            f"- **Likely Causes**: {'; '.join(top_doc.get('root_causes', []))}\n\n"
            f"**Recommended Field Protocol**:\n"
            + "\n".join([f"  {step}" for step in top_doc.get("inspection_procedure", [])[:4]])
            + f"\n\n**Safety / Dispatch**: {top_doc.get('dispatch_protocol', 'Review maintenance logs before flight clearance.')}"
        )

        return {
            "query": question,
            "answer": answer,
            "citations": citations,
            "retrieved_documents": [r["document"] for r in results]
        }
