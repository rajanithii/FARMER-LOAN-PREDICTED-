import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Farmer Loan Default Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #1b4332 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.app-header h1 { color: #d8f3dc; font-size: 2.2rem; margin: 0; font-weight: 700; }
.app-header p  { color: #95d5b2; font-size: 1rem; margin: 0.4rem 0 0 0; }

/* ── Metric cards ── */
.metric-card {
    background: #1e2a1e;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label { color: #95d5b2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #d8f3dc; font-size: 2rem; font-weight: 700; margin-top: 0.3rem; }

/* ── Risk badges ── */
.risk-high     { background:#4a1010; border:1px solid #e74c3c; color:#ff6b6b; padding:1rem 1.5rem; border-radius:12px; font-size:1.1rem; font-weight:700; text-align:center; }
.risk-moderate { background:#3d2b00; border:1px solid #f39c12; color:#ffd166; padding:1rem 1.5rem; border-radius:12px; font-size:1.1rem; font-weight:700; text-align:center; }
.risk-low      { background:#0d2e1a; border:1px solid #2ecc71; color:#55efc4; padding:1rem 1.5rem; border-radius:12px; font-size:1.1rem; font-weight:700; text-align:center; }

/* ── Section headings ── */
.section-title { color: #74c69d; font-size: 1.1rem; font-weight: 600; margin: 1.2rem 0 0.6rem 0; letter-spacing: 0.5px; }

/* ── Recommendation cards ── */
.rec-card { background:#1a2e1a; border-left:4px solid #40916c; border-radius:8px; padding:0.8rem 1rem; margin:0.5rem 0; color:#b7e4c7; font-size:0.92rem; }

/* ── History table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.5rem 1.2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d1f0d; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label { color: #95d5b2 !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Encoding constants (must match LabelEncoder alphabetical order in train_model.py) ──
CROP_CLASSES       = sorted(['Rice','Wheat','Cotton','Maize','Sugarcane','Groundnut','Soybean','Vegetables','Coconut'])
REPAY_CLASSES      = sorted(['Good','Poor'])
SOIL_CLASSES       = sorted(['High','Medium','Low'])
IRRIGATION_CLASSES = sorted(['Canal','Borewell','Drip','Rainfed'])
STATE_CLASSES      = sorted(['Tamil Nadu','Maharashtra','Andhra Pradesh','Punjab','Kerala',
                              'Rajasthan','Madhya Pradesh','Bihar','Telangana','Karnataka',
                              'Gujarat','Haryana','Uttar Pradesh','Odisha','West Bengal'])
FEATURES = [
    "age","land_area_acres","annual_income","crop_type",
    "loan_amount","loan_tenure_months","previous_loans",
    "repayment_history","soil_quality","irrigation_type",
    "credit_score","state","rainfall_mm","avg_temp_celsius"
]
FEATURE_LABELS = {
    "age": "Age", "land_area_acres": "Land Area (acres)",
    "annual_income": "Annual Income", "crop_type": "Crop Type",
    "loan_amount": "Loan Amount", "loan_tenure_months": "Loan Tenure",
    "previous_loans": "Previous Loans", "repayment_history": "Repayment History",
    "soil_quality": "Soil Quality", "irrigation_type": "Irrigation Type",
    "credit_score": "Credit Score", "state": "State",
    "rainfall_mm": "Annual Rainfall", "avg_temp_celsius": "Avg Temperature"
}

def label_encode(value, classes):
    return classes.index(value)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
    return joblib.load(model_path)

model = load_model()
feature_importances = model.feature_importances_

# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🌾 Farmer Loan Default Predictor</h1>
  <p>AI-powered risk assessment for agricultural borrowers — powered by Random Forest & explainability analytics</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown("---")

    st.markdown("**👤 Personal & Financial**")
    age            = st.slider("Age (years)", 18, 75, 40)
    land_area      = st.number_input("Land Area (acres)", 0.1, 50.0, 5.0, 0.1)
    annual_income  = st.number_input("Annual Income (₹)", 5000, 500000, 75000, 1000)
    credit_score   = st.slider("Credit Score", 300, 850, 650)
    previous_loans = st.slider("Previous Loans", 0, 10, 1)
    repayment      = st.selectbox("Repayment History", REPAY_CLASSES)

    st.markdown("**🌾 Agricultural**")
    crop_type      = st.selectbox("Crop Type", CROP_CLASSES)
    soil_quality   = st.selectbox("Soil Quality", SOIL_CLASSES)
    irrigation     = st.selectbox("Irrigation Type", IRRIGATION_CLASSES)
    state          = st.selectbox("State", STATE_CLASSES)

    st.markdown("**💰 Loan Details**")
    loan_amount    = st.number_input("Loan Amount (₹)", 5000, 500000, 50000, 1000)
    loan_tenure    = st.selectbox("Tenure (months)", [12, 18, 24, 30, 36, 48])

    st.markdown("**🌦️ Weather Data**")
    rainfall       = st.slider("Annual Rainfall (mm)", 200, 3000, 800)
    avg_temp       = st.slider("Avg Temperature (°C)", 15, 45, 28)

# ── Helper: encode inputs ─────────────────────────────────────────────────────
def build_input(ag=age, la=land_area, ai=annual_income, ct=crop_type,
                lamt=loan_amount, lt=loan_tenure, pl=previous_loans,
                rh=repayment, sq=soil_quality, it=irrigation,
                cs=credit_score, st_=state, rf=rainfall, tmp=avg_temp):
    return pd.DataFrame([{
        "age": ag, "land_area_acres": la, "annual_income": ai,
        "crop_type":          label_encode(ct,  CROP_CLASSES),
        "loan_amount": lamt,  "loan_tenure_months": lt,
        "previous_loans": pl,
        "repayment_history":  label_encode(rh,  REPAY_CLASSES),
        "soil_quality":       label_encode(sq,  SOIL_CLASSES),
        "irrigation_type":    label_encode(it,  IRRIGATION_CLASSES),
        "credit_score": cs,
        "state":              label_encode(st_, STATE_CLASSES),
        "rainfall_mm": rf,    "avg_temp_celsius": tmp,
    }])

def predict(df):
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return pred, proba

# ── Helper: feature explanation (manual SHAP-like) ────────────────────────────
def explain(input_df):
    """
    Approximates feature contribution using model feature importances
    weighted by the normalised deviation of each input from the dataset median.
    Install `shap` (pip install shap) and replace this with shap.TreeExplainer
    for true SHAP values.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "loan_data.csv")
    df_ref = pd.read_csv(data_path)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['crop_type','repayment_history','soil_quality','irrigation_type','state']:
        df_ref[col] = le.fit_transform(df_ref[col])
    medians = df_ref[FEATURES].median()
    stds    = df_ref[FEATURES].std().replace(0, 1)
    row     = input_df.iloc[0]
    deviations = ((row - medians) / stds).abs()
    contributions = feature_importances * deviations.values
    result = pd.Series(contributions, index=FEATURES)
    return result.sort_values(ascending=False)

# ── Helper: recommendation engine ────────────────────────────────────────────
def recommendations(pred, proba, input_df):
    recs = []
    risk = round(proba[1] * 100, 1)
    row  = input_df.iloc[0]

    if pred == 1:
        if row["credit_score"] < 650:
            recs.append("📈 Improve credit score above 650 — this is the strongest signal for approval.")
        if row["previous_loans"] >= 3:
            recs.append("🔄 Reduce outstanding loan count before applying — high prior debt raises risk significantly.")
        if label_encode(repayment, REPAY_CLASSES) == 1:
            recs.append("✅ Demonstrate consistent repayment on existing loans before requesting new credit.")
        if row["loan_amount"] > row["annual_income"] * 0.8:
            suggested = int(row["annual_income"] * 0.6)
            recs.append(f"💰 Consider reducing loan amount to ₹{suggested:,} — current amount is high relative to income.")
        if label_encode(irrigation, IRRIGATION_CLASSES) == IRRIGATION_CLASSES.index("Rainfed"):
            recs.append("🌊 Switching to canal or drip irrigation improves agricultural stability score.")
        if label_encode(soil_quality, SOIL_CLASSES) == SOIL_CLASSES.index("Low"):
            recs.append("🌱 Low soil quality increases perceived agricultural risk — consider soil improvement documentation.")
        if not recs:
            recs.append("⚠️ Multiple moderate risk factors detected. Strengthening credit history is the most impactful action.")
    else:
        recs.append(f"✅ Profile is strong. Default probability is only {risk}%.")
        if risk > 15:
            recs.append("📋 Consider providing additional collateral documentation to further strengthen the application.")
        recs.append("🏦 Eligible for standard loan processing — no high-risk flags detected.")

    return recs

# ── Helper: plotly gauge ──────────────────────────────────────────────────────
def risk_gauge(risk_score):
    color = "#e74c3c" if risk_score >= 60 else "#f39c12" if risk_score >= 30 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#aaa"}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a1a",
            "steps": [
                {"range": [0,  30], "color": "#0d2e1a"},
                {"range": [30, 60], "color": "#2d1f00"},
                {"range": [60,100], "color": "#2e0d0d"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": risk_score}
        },
        title={"text": "Default Risk Score", "font": {"color": "#95d5b2", "size": 14}}
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc"
    )
    return fig

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📁 Batch Prediction",
    "📊 Prediction History",
    "🔄 What-If Simulator"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_btn, _, _ = st.columns([1, 1, 1])
    with col_btn:
        predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True, type="primary")

    if predict_btn:
        input_df       = build_input()
        pred, proba    = predict(input_df)
        risk_score     = round(proba[1] * 100, 1)
        safe_score     = round(proba[0] * 100, 1)

        # ── Risk badge ──
        st.markdown("<br>", unsafe_allow_html=True)
        if risk_score >= 60:
            st.markdown(f'<div class="risk-high">⚠️ HIGH RISK — Likely to Default &nbsp;|&nbsp; {risk_score}% probability</div>', unsafe_allow_html=True)
        elif risk_score >= 30:
            st.markdown(f'<div class="risk-moderate">⚡ MODERATE RISK — Monitor Closely &nbsp;|&nbsp; {risk_score}% probability</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK — Unlikely to Default &nbsp;|&nbsp; {risk_score}% probability</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Gauge + Metrics ──
        g1, g2, g3, g4 = st.columns([2, 1, 1, 1])
        with g1:
            st.plotly_chart(risk_gauge(risk_score), use_container_width=True)
        with g2:
            st.markdown(f'<div class="metric-card"><div class="label">Default Risk</div><div class="value" style="color:#e74c3c">{risk_score}%</div></div>', unsafe_allow_html=True)
        with g3:
            st.markdown(f'<div class="metric-card"><div class="label">Safe Probability</div><div class="value" style="color:#2ecc71">{safe_score}%</div></div>', unsafe_allow_html=True)
        with g4:
            level = "High" if risk_score >= 60 else "Moderate" if risk_score >= 30 else "Low"
            level_color = "#e74c3c" if risk_score >= 60 else "#f39c12" if risk_score >= 30 else "#2ecc71"
            st.markdown(f'<div class="metric-card"><div class="label">Risk Level</div><div class="value" style="color:{level_color};font-size:1.4rem">{level}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Feature Explainer + Recommendations ──
        ex_col, rec_col = st.columns([1.4, 1])

        with ex_col:
            st.markdown('<div class="section-title">📊 Feature Impact Analysis</div>', unsafe_allow_html=True)
            contribs = explain(input_df)
            top = contribs.head(8)
            labels = [FEATURE_LABELS.get(f, f) for f in top.index]
            colors = [
                "#e74c3c" if v > top.quantile(0.66)
                else "#f39c12" if v > top.quantile(0.33)
                else "#2ecc71"
                for v in top.values
            ]
            fig_bar = go.Figure(go.Bar(
                x=top.values[::-1], y=labels[::-1],
                orientation='h',
                marker_color=colors[::-1],
                text=[f"{v:.4f}" for v in top.values[::-1]],
                textposition='outside',
                textfont=dict(color="#ccc", size=11)
            ))
            fig_bar.update_layout(
                height=320, margin=dict(t=10, b=10, l=10, r=60),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#2a2a2a", color="#888", title="Contribution Score"),
                yaxis=dict(color="#95d5b2"),
                font_color="#ccc"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Red = highest risk contribution · Orange = moderate · Green = low contribution")

        with rec_col:
            st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)
            recs = recommendations(pred, proba, input_df)
            for r in recs:
                st.markdown(f'<div class="rec-card">{r}</div>', unsafe_allow_html=True)

        # ── Save to history ──
        st.session_state.history.append({
            "Time":         datetime.now().strftime("%H:%M:%S"),
            "Applicant Age": age,
            "Credit Score": credit_score,
            "Loan (₹)":     f"₹{loan_amount:,}",
            "Income (₹)":   f"₹{annual_income:,}",
            "Crop":         crop_type,
            "State":        state,
            "Risk Score":   f"{risk_score}%",
            "Verdict":      "⚠️ High" if risk_score >= 60 else "⚡ Moderate" if risk_score >= 30 else "✅ Low"
        })

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📁 Batch Loan Risk Assessment")
    st.markdown(
        "Upload a CSV file containing multiple applicant records. "
        "The file must include these columns: "
        "`age, land_area_acres, annual_income, crop_type, loan_amount, "
        "loan_tenure_months, previous_loans, repayment_history, soil_quality, "
        "irrigation_type, credit_score, state, rainfall_mm, avg_temp_celsius`"
    )

    # ── Download template ──
    template_df = pd.DataFrame([{
        "age": 40, "land_area_acres": 5.0, "annual_income": 75000,
        "crop_type": "Rice", "loan_amount": 50000, "loan_tenure_months": 24,
        "previous_loans": 1, "repayment_history": "Good",
        "soil_quality": "High", "irrigation_type": "Canal",
        "credit_score": 680, "state": "Tamil Nadu",
        "rainfall_mm": 950, "avg_temp_celsius": 28
    }])
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Template", csv_template, "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(batch_df)} records successfully.")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                batch_encoded = batch_df.copy()
                for col, classes in [
                    ("crop_type", CROP_CLASSES), ("repayment_history", REPAY_CLASSES),
                    ("soil_quality", SOIL_CLASSES), ("irrigation_type", IRRIGATION_CLASSES),
                    ("state", STATE_CLASSES)
                ]:
                    batch_encoded[col] = batch_encoded[col].apply(
                        lambda v: classes.index(v) if v in classes else 0
                    )

                X_batch  = batch_encoded[FEATURES]
                preds    = model.predict(X_batch)
                probas   = model.predict_proba(X_batch)[:, 1]

                results           = batch_df.copy()
                results["Risk %"] = (probas * 100).round(1)
                results["Verdict"] = [
                    "⚠️ High" if p >= 60 else "⚡ Moderate" if p >= 30 else "✅ Low"
                    for p in (probas * 100)
                ]

                st.markdown("### 📊 Batch Results")
                st.dataframe(results, use_container_width=True)

                # Summary metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Applicants",   len(results))
                m2.metric("High Risk Count",    int((probas >= 0.6).sum()))
                m3.metric("Average Risk Score", f"{(probas*100).mean():.1f}%")

                # Risk distribution chart
                fig_dist = px.histogram(
                    results, x="Risk %", nbins=20,
                    title="Risk Score Distribution",
                    color_discrete_sequence=["#40916c"]
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    xaxis=dict(gridcolor="#2a2a2a"),
                    yaxis=dict(gridcolor="#2a2a2a")
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # Download results
                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results CSV",
                    csv_out, "batch_results.csv", "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Prediction History
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Session Prediction History")
    if not st.session_state.history:
        st.info("No predictions made yet in this session. Go to **Single Prediction** and run a prediction.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Summary
        total  = len(history_df)
        high   = history_df["Verdict"].str.contains("High").sum()
        low    = history_df["Verdict"].str.contains("Low").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", total)
        c2.metric("High Risk",         high)
        c3.metric("Low Risk",          low)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        # Download history
        hist_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download History", hist_csv, "prediction_history.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — What-If Simulator
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔄 What-If Simulator")
    st.markdown(
        "Adjust any factor below to see in real time how it affects the default risk score. "
        "This helps identify which changes would most improve a loan application."
    )

    w1, w2 = st.columns(2)

    with w1:
        st.markdown("**Adjust Parameters**")
        wi_credit   = st.slider("Credit Score",          300,  850,  credit_score,  key="wi_cs")
        wi_income   = st.slider("Annual Income (₹)",     5000, 500000, annual_income, 5000, key="wi_ai")
        wi_loan     = st.slider("Loan Amount (₹)",       5000, 500000, loan_amount,   5000, key="wi_la")
        wi_prevloan = st.slider("Previous Loans",        0,    10,   previous_loans,       key="wi_pl")
        wi_rainfall = st.slider("Annual Rainfall (mm)",  200,  3000, rainfall,             key="wi_rf")
        wi_repay    = st.selectbox("Repayment History",  REPAY_CLASSES, key="wi_rh")
        wi_soil     = st.selectbox("Soil Quality",       SOIL_CLASSES,  key="wi_sq")
        wi_irr      = st.selectbox("Irrigation Type",    IRRIGATION_CLASSES, key="wi_ir")

    with w2:
        st.markdown("**Live Risk Output**")

        wi_input = build_input(
            ag=age, la=land_area, ai=wi_income, ct=crop_type,
            lamt=wi_loan, lt=loan_tenure, pl=wi_prevloan,
            rh=wi_repay, sq=wi_soil, it=wi_irr,
            cs=wi_credit, st_=state, rf=wi_rainfall, tmp=avg_temp
        )
        wi_pred, wi_proba = predict(wi_input)
        wi_risk = round(wi_proba[1] * 100, 1)

        # Live gauge
        st.plotly_chart(risk_gauge(wi_risk), use_container_width=True)

        # Delta vs original prediction
        orig_input       = build_input()
        _, orig_proba    = predict(orig_input)
        orig_risk        = round(orig_proba[1] * 100, 1)
        delta            = round(wi_risk - orig_risk, 1)
        delta_str        = f"+{delta}%" if delta > 0 else f"{delta}%"
        delta_color      = "#e74c3c" if delta > 0 else "#2ecc71" if delta < 0 else "#888"

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Change from Original</div>'
            f'<div class="value" style="color:{delta_color}">{delta_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if wi_risk < orig_risk:
            st.success(f"✅ These changes reduce risk by {abs(delta):.1f}% — a stronger application.")
        elif wi_risk > orig_risk:
            st.error(f"⚠️ These changes increase risk by {delta:.1f}% — application weakens.")
        else:
            st.info("No change in risk score with current adjustments.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Developed by Rajanithi N · AI & Data Science · DSU-SET · farmer-loan-predictor v2.0")
