"""
Streamlit Web Dashboard for NASA CMAPSS Predictive Maintenance & RAG Copilot.
Features:
- Fleet Telemetry Monitoring & Risk Matrix
- Engine Unit Deep Dive & Degradation Curves
- RAG Intelligent Maintenance Copilot & Work Order Generation
- MLOps Model Performance & Diagnostic Auditing
"""

import sys
from pathlib import Path

# Add project root to sys.path to support execution from any directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="AeroGuard | Predictive Maintenance & RAG Copilot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import pipeline & RAG components
from src.pipelines.inference_pipeline import PredictiveMaintenanceInferenceService
from src.rag.engine import MaintenanceCopilot


@st.cache_resource
def load_services():
    inference_svc = PredictiveMaintenanceInferenceService(root_dir=ROOT)
    copilot_svc = MaintenanceCopilot(kb_dir=ROOT / "src" / "rag" / "knowledge_base")
    return inference_svc, copilot_svc


@st.cache_data
def load_dataset():
    feat_path = ROOT / "data" / "03-features" / "FD001_features.csv"
    if not feat_path.exists():
        prep_path = ROOT / "data" / "02-preprocessed" / "FD001_preprocessed.csv"
        return pd.read_csv(prep_path)
    return pd.read_csv(feat_path)


@st.cache_data
def load_eval_data():
    metrics_path = ROOT / "data" / "03-predictions" / "eval_metrics.json"
    fi_path = ROOT / "data" / "03-predictions" / "feature_importances.csv"
    report_path = ROOT / "data" / "03-predictions" / "eval_report.csv"

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    fi_df = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
    report_df = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
    return metrics, fi_df, report_df


# Load components
inference_svc, copilot = load_services()
full_df = load_dataset()
metrics, fi_df, report_df = load_eval_data()

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/airplane-take-off.png", width=64)
st.sidebar.title("AeroGuard Analytics")
st.sidebar.markdown(
    "**Aviation Turbofan Reliability & Predictive Maintenance System**\n\n"
    "NASA CMAPSS FD001 Telemetry Dataset"
)

st.sidebar.divider()
st.sidebar.markdown("### System Status")
st.sidebar.success("● ML Model: Active (Random Forest)")
st.sidebar.success("● RAG Knowledge Base: 5 Technical SOPs")
st.sidebar.info(f"● Ingested Telemetry: {len(full_df):,} cycles")
st.sidebar.info(f"● Monitored Fleet: {full_df['unit'].nunique()} Engines")

st.sidebar.divider()
st.sidebar.caption("AeroGuard • Machine Learning & Telemetry Analytics")

# -------------------------------------------------------------
# Main Tabs
# -------------------------------------------------------------
st.title("✈️ AeroGuard: Turbofan Predictive Maintenance & RAG Copilot")
st.markdown(
    "Multivariate sensor telemetry degradation modeling, early warning failure classification, "
    "and RAG-driven prescriptive maintenance advisories."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Fleet Telemetry Overview",
    "🔍 Engine Degradation Deep Dive",
    "🤖 AI Maintenance Copilot (RAG)",
    "📈 Model Accuracy & Evaluation Metrics"
])

# =============================================================
# TAB 1: FLEET OPERATIONS CENTER
# =============================================================
with tab1:
    st.subheader("Fleet-Wide Reliability & Failure Risk Matrix")

    # Fleet metrics calculation
    latest_cycles = full_df.loc[full_df.groupby("unit")["time_cycle"].idxmax()].copy()
    
    # Calculate current risk for all engines at latest cycle
    if "predicted_risk" not in latest_cycles.columns and inference_svc.is_ready:
        drop_cols = {"unit", "time_cycle", "max_cycle", "RUL", "failure_within_30"}
        feat_cols = [c for c in latest_cycles.columns if c not in drop_cols]
        X_latest = latest_cycles[feat_cols].values
        if inference_svc.imputer:
            X_latest = inference_svc.imputer.transform(X_latest)
        if inference_svc.scaler:
            X_latest = inference_svc.scaler.transform(X_latest)
        latest_cycles["predicted_risk"] = inference_svc.model.predict_proba(X_latest)[:, 1]
    
    critical_count = (latest_cycles["predicted_risk"] >= 0.70).sum()
    warning_count = ((latest_cycles["predicted_risk"] >= 0.35) & (latest_cycles["predicted_risk"] < 0.70)).sum()
    normal_count = (latest_cycles["predicted_risk"] < 0.35).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Engines", f"{len(latest_cycles)}")
    col2.metric("Critical Alerts (P1)", f"{critical_count}", delta=f"{critical_count} AOG required", delta_color="inverse")
    col3.metric("Warning Alerts (P2)", f"{warning_count}", delta=f"{warning_count} inspect <25c", delta_color="off")
    col4.metric("Normal Fleet (P3)", f"{normal_count}", delta=f"{(normal_count/len(latest_cycles)*100):.1f}% nominal")

    st.markdown("---")

    # Fleet scatter plot
    fig_fleet = px.scatter(
        latest_cycles,
        x="time_cycle",
        y="predicted_risk",
        color="predicted_risk",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        size="time_cycle",
        hover_data=["unit", "time_cycle", "RUL"],
        title="Fleet Telemetry: Operating Cycles vs. Predicted Failure Risk (Within 30 Cycles)",
        labels={"time_cycle": "Operating Cycles Logged", "predicted_risk": "Predicted Failure Risk"},
    )
    fig_fleet.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="Critical Threshold (0.70)")
    fig_fleet.add_hline(y=0.35, line_dash="dash", line_color="orange", annotation_text="Warning Threshold (0.35)")
    fig_fleet.update_layout(height=420)
    st.plotly_chart(fig_fleet, use_container_width=True)

    # Priority Action Alert Table
    st.markdown("### ⚠️ Priority Action Fleet Roster (Highest Risk Units)")
    priority_df = latest_cycles.sort_values("predicted_risk", ascending=False)[
        ["unit", "time_cycle", "RUL", "predicted_risk"]
    ].copy()
    priority_df["risk_level"] = priority_df["predicted_risk"].apply(
        lambda r: "🔴 CRITICAL" if r >= 0.70 else ("🟡 WARNING" if r >= 0.35 else "🟢 NORMAL")
    )
    priority_df["predicted_risk"] = priority_df["predicted_risk"].apply(lambda r: f"{r*100:.1f}%")
    st.dataframe(
        priority_df.head(10),
        column_config={
            "unit": "Engine Unit #",
            "time_cycle": "Current Cycle",
            "RUL": "Remaining Useful Life (Cycles)",
            "predicted_risk": "Failure Risk Probability",
            "risk_level": "Dispatch Status"
        },
        use_container_width=True,
        hide_index=True
    )


# =============================================================
# TAB 2: ENGINE UNIT DEEP DIVE
# =============================================================
with tab2:
    st.subheader("Individual Engine Telemetry & Degradation Trajectory")

    unit_ids = sorted(full_df["unit"].unique())
    selected_unit = st.selectbox("Select Engine Unit ID to Inspect:", unit_ids, index=0)

    unit_data = inference_svc.load_unit_history(selected_unit)

    if unit_data is not None:
        max_c = int(unit_data["time_cycle"].max())
        selected_cycle = st.slider(
            "Telemetry Observation Cycle:",
            min_value=1,
            max_value=max_c,
            value=max_c,
            help="Slide across the engine's lifetime to observe degradation dynamics."
        )

        current_row = unit_data[unit_data["time_cycle"] == selected_cycle].iloc[0]
        cur_risk = float(current_row["predicted_risk"])
        cur_level = str(current_row["risk_level"])
        cur_rul = int(current_row["RUL"]) if "RUL" in current_row else None

        # Display snapshot metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Cycle", f"{selected_cycle} of {max_c}")
        c2.metric("Failure Risk (within 30c)", f"{cur_risk*100:.1f}%")
        c3.metric("Status Level", cur_level)
        if cur_rul is not None:
            c4.metric("True RUL", f"{cur_rul} cycles")

        # Two columns: Sensor degradation curves + Risk curve
        row_col1, row_col2 = st.columns(2)

        with row_col1:
            # Key sensors plot
            fig_sensors = go.Figure()
            # Informative sensors: sensor_11 (HPC static pressure), sensor_4 (LPT temp), sensor_15 (Bypass ratio)
            fig_sensors.add_trace(go.Scatter(
                x=unit_data["time_cycle"], y=unit_data["sensor_11"],
                mode="lines", name="S11: HPC Static Pressure (psia)", line=dict(color="#3B82F6", width=2)
            ))
            fig_sensors.add_trace(go.Scatter(
                x=unit_data["time_cycle"], y=unit_data["sensor_4"],
                mode="lines", name="S4: LPT Outlet Temp (°R)", line=dict(color="#EF4444", width=2), yaxis="y2"
            ))
            fig_sensors.add_vline(x=selected_cycle, line_dash="dot", line_color="gray", annotation_text=f"Cycle {selected_cycle}")
            fig_sensors.update_layout(
                title=f"Unit #{selected_unit}: Key Sensor Trajectories Across Cycles",
                xaxis=dict(title="Operating Cycle"),
                yaxis=dict(title="Sensor 11 (psia)", side="left"),
                yaxis2=dict(title="Sensor 4 (°R)", side="right", overlaying="y"),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sensors, use_container_width=True)

        with row_col2:
            # Risk curve
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=unit_data["time_cycle"], y=unit_data["predicted_risk"],
                mode="lines+markers", name="Failure Risk", line=dict(color="#8B5CF6", width=3)
            ))
            fig_risk.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="Critical (0.70)")
            fig_risk.add_hline(y=0.35, line_dash="dash", line_color="orange", annotation_text="Warning (0.35)")
            fig_risk.add_vline(x=selected_cycle, line_dash="dot", line_color="gray")
            fig_risk.update_layout(
                title=f"Unit #{selected_unit}: Predicted Failure Probability Trajectory",
                xaxis=dict(title="Operating Cycle"),
                yaxis=dict(title="Failure Risk Probability", range=[0, 1.05]),
                height=380
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        # Anomaly attribution for the selected cycle
        st.markdown("#### Primary Telemetry Anomalies at Selected Cycle")
        drop_cols = {"unit", "time_cycle", "max_cycle", "RUL", "failure_within_30", "predicted_risk", "predicted_label", "risk_level"}
        feat_dict = {c: float(current_row[c]) for c in current_row.index if c not in drop_cols and pd.notnull(current_row[c])}
        pred_res = inference_svc.predict_from_features(feat_dict)

        top_factors = pred_res.get("top_degrading_factors", [])
        if top_factors:
            f_cols = st.columns(len(top_factors))
            for i, factor in enumerate(top_factors):
                with f_cols[i]:
                    st.metric(
                        label=f"{factor['sensor']}",
                        value=f"{factor['z_score']:+.2f} σ",
                        delta=f"{factor['deviation_direction']} ({factor['severity']})",
                        delta_color="inverse" if factor["severity"] == "HIGH" else "off"
                    )


# =============================================================
# TAB 3: AI MAINTENANCE COPILOT (RAG)
# =============================================================
with tab3:
    st.subheader("RAG Prescriptive Maintenance Copilot & Work-Order Generator")
    st.markdown(
        "Combines real-time telemetry degradation alerts with authentic NASA & OEM technical manuals, "
        "Standard Operating Procedures (SOP), and root cause analysis."
    )

    rag_col1, rag_col2 = st.columns([1, 1.4])

    with rag_col1:
        st.markdown("#### Telemetry Incident Context")
        eng_id = st.number_input("Target Engine Unit ID:", min_value=1, max_value=100, value=selected_unit)
        
        # Pull latest telemetry for that unit
        unit_history = inference_svc.load_unit_history(eng_id)
        if unit_history is not None:
            latest_row = unit_history.iloc[-1]
            default_cycle = int(latest_row["time_cycle"])
            default_risk = float(latest_row["predicted_risk"])
            default_level = str(latest_row["risk_level"])
        else:
            default_cycle = 150
            default_risk = 0.85
            default_level = "CRITICAL"

        sim_cycle = st.slider("Cycle Number:", min_value=1, max_value=400, value=default_cycle)
        sim_risk = st.slider("Simulated Failure Risk:", min_value=0.0, max_value=1.0, value=default_risk, step=0.01)
        sim_level = "CRITICAL" if sim_risk >= 0.70 else ("WARNING" if sim_risk >= 0.35 else "NORMAL")
        st.info(f"Classified Risk State: **{sim_level}**")

        st.markdown("#### Primary Degrading Telemetry Signals")
        anom_sensor = st.multiselect(
            "Active Degraded Sensors:",
            ["sensor_11 (HPC Static Pressure)", "sensor_4 (LPT Outlet Temp)", "sensor_3 (HPC Total Pressure)", "sensor_9 (Core Speed)", "sensor_15 (Bypass Ratio)", "sensor_17 (Bleed Enthalpy)"],
            default=["sensor_11 (HPC Static Pressure)", "sensor_4 (LPT Outlet Temp)"]
        )

        sensor_tags = [s.split()[0] for s in anom_sensor]
        factors = [{"sensor": s, "deviation_direction": "ABNORMAL"} for s in sensor_tags]

        generate_btn = st.button("🛠️ Generate Technical Work Order (RAG)", type="primary", use_container_width=True)

    with rag_col2:
        st.markdown("#### Prescriptive Maintenance Work-Order")
        if generate_btn or "last_wo" in st.session_state:
            if generate_btn:
                wo = copilot.generate_work_order(
                    unit_id=eng_id,
                    cycle=sim_cycle,
                    failure_risk=sim_risk,
                    risk_level=sim_level,
                    degrading_factors=factors
                )
                st.session_state["last_wo"] = wo
            else:
                wo = st.session_state["last_wo"]

            st.success(f"**Work Order ID**: `{wo['work_order_id']}` | Priority: **{wo['dispatch_priority']}**")
            st.markdown(f"**Generated**: `{wo['timestamp']}`")
            st.markdown(f"**Diagnostic Summary**: {wo['telemetry_summary']}")

            # Matched SOPs
            st.markdown("##### 📄 Matched Technical SOPs & Orders:")
            for sop in wo["matched_sops"]:
                st.markdown(f"- **{sop['doc_id']}**: {sop['title']} *(Subsystem: {sop['subsystem']} | Relevance: {sop['relevance_score']})*")

            # Root Causes
            st.markdown("##### 🔬 Root Cause Hypotheses:")
            for rc in wo["identified_root_causes"]:
                st.markdown(f"- {rc}")

            # Mandatory Field Checklist
            st.markdown("##### 📋 Field Action Protocol Checklist:")
            for step in wo["mandatory_action_checklist"]:
                st.checkbox(step, key=f"step_{step[:15]}")

            # Parts required
            st.markdown("##### 🔩 Required Spare Parts & Tooling:")
            parts_df = pd.DataFrame(wo["recommended_parts"])
            st.table(parts_df)

            # Safety advisory
            st.warning(f"⚠️ **Safety Directive**: {wo['safety_protocol']}")

    st.divider()

    # Interactive Maintenance Q&A
    st.markdown("### 💬 Ask the AeroGuard Technical Knowledge Base")
    user_query = st.text_input(
        "Enter technical maintenance question or symptom query:",
        placeholder="e.g. What are the inspection steps when sensor 11 static pressure drops below 47 psia?"
    )
    if st.button("Search Maintenance Knowledge Base"):
        if user_query.strip():
            with st.spinner("Retrieving SOPs and synthesizing guidance..."):
                response = copilot.answer_query(user_query)
                st.markdown(response["answer"])
                st.markdown("**Citations:**")
                for cite in response["citations"]:
                    st.caption(f"• {cite}")


# =============================================================
# TAB 4: MODEL ACCURACY & EVALUATION METRICS
# =============================================================
with tab4:
    st.subheader("Model Performance & Accuracy Metrics")
    st.markdown(
        "Evaluation metrics obtained from held-out validation engines using **GroupShuffleSplit** "
        "(zero data leakage across engine units)."
    )

    if metrics:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        m2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        m3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        m4.metric("F1-Score", f"{metrics.get('f1', 0)*100:.2f}%")
        m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        m6.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")

        st.markdown("---")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.markdown("#### Confusion Matrix Heatmap")
            cm_dict = metrics.get("confusion_matrix", {})
            tn = cm_dict.get("true_negatives", 0)
            fp = cm_dict.get("false_positives", 0)
            fn = cm_dict.get("false_negatives", 0)
            tp = cm_dict.get("true_positives", 0)

            cm_matrix = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="Actual Class", color="Sample Count"),
                x=["Healthy (>30c)", "Failing (<=30c)"],
                y=["Healthy (>30c)", "Failing (<=30c)"],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Validation Confusion Matrix (4,070 Cycles Across 20 Held-out Engines)"
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.caption(
                f"**Key Operational Takeaway**: With High Recall of **{metrics.get('recall', 0)*100:.1f}%**, "
                f"only {fn} failing cycles were missed out of {fn+tp} failing cycles, "
                f"while maintaining a low false alarm rate of {fp/(tn+fp)*100:.2f}%."
            )

        with perf_col2:
            st.markdown("#### Top 12 Informative Degradation Features")
            if not fi_df.empty:
                top_fi = fi_df.head(12).sort_values("importance", ascending=True)
                fig_fi = px.bar(
                    top_fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Feature Importance (Random Forest Gini Impurity Reduction)",
                    labels={"importance": "Relative Importance", "feature": "Engineered Telemetry Feature"},
                    color="importance",
                    color_continuous_scale="Purples"
                )
                fig_fi.update_layout(height=380)
                st.plotly_chart(fig_fi, use_container_width=True)
