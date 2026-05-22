"""
InverterShield Streamlit Dashboard.

Run:
    streamlit run src/dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE.parent.parent
_METRICS = _ROOT / "models" / "metrics.json"
_OUTPUTS = _ROOT / "outputs"

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "InverterShield",
    page_icon  = "⚡",
    layout     = "wide",
)

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("⚡ InverterShield")
page = st.sidebar.radio("Navigation", ["Predict", "Model Performance", "SHAP Analysis"])


# ── Predict page ───────────────────────────────────────────────────
if page == "Predict":
    st.title("Inverter Risk Prediction")
    st.markdown("Enter inverter telemetry to get a 7-day failure risk assessment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Inverter")
        inv_power    = st.number_input("Power (kW)",          value=3.9,   step=0.1)
        inv_temp     = st.number_input("Temp (°C)",           value=29.0,  step=0.5)
        inv_freq     = st.number_input("Frequency (Hz)",      value=50.0,  step=0.01)
        inv_kwh_today = st.number_input("Energy today (kWh)", value=437.0, step=1.0)
        inv_pv1_power = st.number_input("PV1 Power (kW)",     value=1.5,   step=0.1)

    with col2:
        st.subheader("Meter / Grid")
        meter_pf   = st.number_input("Power Factor",   value=0.98,  step=0.01, min_value=0.0, max_value=1.0)
        meter_freq = st.number_input("Grid Freq (Hz)", value=50.0,  step=0.01)
        meter_v_r  = st.number_input("Voltage R (V)",  value=236.0, step=1.0)
        meter_v_y  = st.number_input("Voltage Y (V)",  value=236.0, step=1.0)
        meter_v_b  = st.number_input("Voltage B (V)",  value=236.0, step=1.0)

    with col3:
        st.subheader("Alarms / SMU")
        alarm_24h  = st.number_input("Alarms (24h)",           value=0,    step=1)
        alarm_7d   = st.number_input("Alarms (7d)",            value=0,    step=1)
        hrs_alarm  = st.number_input("Hours since last alarm",  value=9999, step=1)
        smu_mean   = st.number_input("String current mean (A)", value=0.013, step=0.001, format="%.4f")
        smu_zero   = st.number_input("Zero-current strings",    value=18,   step=1)
        smu_total  = st.number_input("Total strings",           value=24,   step=1)

    inverter_id = st.text_input("Inverter ID (optional)", value="INV-001")
    ambient     = st.number_input("Ambient Temp (°C)", value=28.0, step=0.5)

    if st.button("Predict Risk", type="primary"):
        payload = {
            "features": {
                "inv_power":             inv_power,
                "inv_temp":              inv_temp,
                "inv_freq":              inv_freq,
                "inv_kwh_today":         inv_kwh_today,
                "inv_pv1_power":         inv_pv1_power,
                "meter_pf":              meter_pf,
                "meter_freq":            meter_freq,
                "meter_v_r":             meter_v_r,
                "meter_v_y":             meter_v_y,
                "meter_v_b":             meter_v_b,
                "alarm_count_24h":       alarm_24h,
                "alarm_count_7d":        alarm_7d,
                "hours_since_last_alarm": hrs_alarm,
                "smu_string_mean":       smu_mean,
                "smu_num_zero":          smu_zero,
                "smu_total_strings":     smu_total,
                "ambient_temp":          ambient,
            },
            "inverter_id": inverter_id,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("API not reachable. Start the API with: uvicorn src.api.main:app --port 8000")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        risk = result["risk_level"]
        prob = result["failure_probability"]

        color_map = {"no_risk": "green", "degradation_risk": "orange", "shutdown_risk": "red"}
        color     = color_map.get(risk, "gray")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level",           risk.replace("_", " ").title())
        c2.metric("Failure Probability",  f"{prob:.1%}")
        c3.metric("Anomaly Detected",     "Yes" if result["is_anomaly"] else "No")

        st.markdown(f"**Recommendation:** {result['recommendation']}")

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = prob * 100,
            title = {"text": "Failure Probability (%)"},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  30], "color": "#d4edda"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100],"color": "#f8d7da"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Top risk factors
        st.subheader("Top Risk Factors")
        factors_df = pd.DataFrame(result["top_risk_factors"])
        fig2 = px.bar(
            factors_df,
            x      = "impact",
            y      = "feature",
            orientation = "h",
            color  = "direction",
            color_discrete_map = {
                "increases risk": "#dc3545",
                "decreases risk": "#28a745",
            },
            labels = {"impact": "SHAP Impact", "feature": "Feature"},
            title  = "SHAP Feature Contributions",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)


# ── Model Performance page ─────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")

    if not _METRICS.exists():
        st.warning("No metrics found. Train the model first with: python run_pipeline.py")
        st.stop()

    metrics = json.loads(_METRICS.read_text())

    st.subheader("Hold-out Test Metrics (Binary)")
    tm = metrics.get("test_binary", {})
    cols = st.columns(4)
    cols[0].metric("F1",        f"{tm.get('f1', 0):.3f}")
    cols[1].metric("Precision", f"{tm.get('precision', 0):.3f}")
    cols[2].metric("Recall",    f"{tm.get('recall', 0):.3f}")
    cols[3].metric("ROC-AUC",   f"{tm.get('roc_auc', 0):.3f}")

    st.markdown(f"**Multi-class Weighted F1 (test):** {metrics.get('test_multiclass_f1', 'N/A')}")

    cv = metrics.get("cv_binary", [])
    if cv:
        st.subheader("Cross-Validation Results")
        cv_df = pd.DataFrame(cv)
        fig   = px.line(
            cv_df.melt(id_vars="fold", value_vars=["f1", "precision", "recall", "roc_auc"]),
            x     = "fold",
            y     = "value",
            color = "variable",
            title = "CV Metrics per Fold",
            markers = True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top SHAP Features")
    for i, feat in enumerate(metrics.get("top_shap_features", []), 1):
        st.markdown(f"{i}. `{feat}`")


# ── SHAP Analysis page ─────────────────────────────────────────────
elif page == "SHAP Analysis":
    st.title("SHAP Feature Importance")

    bar_img = _OUTPUTS / "shap_summary.png"
    bee_img = _OUTPUTS / "shap_beeswarm.png"

    if bar_img.exists():
        st.image(str(bar_img), caption="Mean |SHAP| — Top 10 Features", use_container_width=True)
    else:
        st.info("SHAP bar chart not found. Run run_pipeline.py to generate it.")

    if bee_img.exists():
        st.image(str(bee_img), caption="SHAP Beeswarm Plot", use_container_width=True)
