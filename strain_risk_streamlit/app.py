import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
import tempfile

# ---------------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ---------------------------------------------------------
st.set_page_config(layout="wide")

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Target tab buttons */
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 32px !important;
        font-weight: 700 !important;
    }

    /* Make active tab even stronger */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        font-size: 32px !important;
        font-weight: 800 !important;
    }

    /* Space under tabs */
    div[data-testid="stTabs"] {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("School Strain Monitoring Platform")

# ---------------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------------
DATA_URL = "https://huggingface.co/datasets/whitfieldscott/school_strain_data/resolve/main/xgb_production_dataset.csv"
MODEL_URL = "https://huggingface.co/datasets/whitfieldscott/school_strain_data/resolve/main/xgb_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

@st.cache_resource
def load_model_and_features(dataframe: pd.DataFrame):
    """
    Load XGBoost model and derive:
    
    - feature_list from dataframe (drop ID/target columns)
    - feature_importance_df from model.get_booster().get_score('gain')
      aligned to feature_list so it always matches the dataset columns.
    """

    # Download model from HuggingFace
    response = requests.get(MODEL_URL)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    model = joblib.load(tmp_path)

    # Build feature list from the dataset (most reliable)
    exclude_cols = {"NCESSCH", "SCH_NAME", "SURVYEAR", "high_strain"}
    feature_list = [c for c in dataframe.columns if c not in exclude_cols]

    # Compute feature importance from model
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")  # dict: feature -> gain

    # Align importance to dataset features
    feature_importance_df = (
        pd.DataFrame({"feature": feature_list})
        .assign(total_gain=lambda d: d["feature"].map(gain_scores).fillna(0.0))
        .sort_values("total_gain", ascending=False)
        .reset_index(drop=True)
    )

    total = feature_importance_df["total_gain"].sum()
    feature_importance_df["pct_of_total"] = (
        feature_importance_df["total_gain"] / total if total > 0 else 0.0
    )

    return model, feature_list, feature_importance_df

model, feature_list, feature_importance_df = load_model_and_features(df)

# ---------------------------------------------------------
# CREATE TABS (TAB 1 + TAB 2 ONLY)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Project Overview", "School Deep Dive"])

# =========================================================
# TAB 1 — PROJECT OVERVIEW
# =========================================================
with tab1:

    # -----------------------------------------------------
    # TITLE + POSITIONING
    # -----------------------------------------------------
    st.header("School Strain Monitoring Platform")

    st.markdown("""
    ### Research Question

    Can socioeconomic and demographic characteristics be used to predict  
    which schools are at risk for elevated student–teacher structural strain?

    ---
    """)

    st.markdown("""
    ### Project Framing

    A data-driven early warning system designed to identify schools  
    at elevated risk of structural strain before operational pressures escalate.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # BUSINESS PROBLEM
    # =====================================================
    st.subheader("Business Problem")

    st.markdown("""
    • Structural strain impacts instructional quality and staffing balance  

    • Overcrowding pressures are often addressed reactively rather than proactively  

    • District leadership lacks a consistent early-warning framework  

    • Resource allocation decisions require prioritization across schools  
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # DATA COLLECTION & SCOPE
    # =====================================================
    st.subheader("Data Collection & Scope")

    st.markdown("""
    • **Data Source:** NCES school-level + socioeconomic census indicators  

    • **Time Span:** 2017–2022  

    • **Target Variable:** High Strain (binary classification)  
      Defined using elevated student–teacher ratio thresholds  

    • **Modeling Objective:** Predict structural strain risk using  
      socioeconomic and demographic indicators  

    • **Models Used:** Random Forest & XGBoost  

    • **Evaluation Metrics:** ROC-AUC and Precision-Recall  

    ⚠️ This project evaluates predictive capability — not causal proof.
    """)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # =====================================================
    # KEY SYSTEM-WIDE FINDINGS
    # =====================================================
    st.subheader("Key Findings")

    strain_rate = df["high_strain"].mean()
    st.metric("Overall High Strain Rate Across Schools", f"{strain_rate:.2%}")

    X_all = df[feature_list]
    all_probs = model.predict_proba(X_all)[:, 1]

    col1, col2 = st.columns(2)

    # ---------------- Risk Distribution ----------------
    with col1:
        st.markdown("**System-Wide Risk Distribution**")

        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        ax.hist(all_probs, bins=30)
        ax.set_xlabel("Predicted Risk", fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        Most schools cluster at lower predicted risk levels,  
        while a smaller segment consistently shows elevated strain probability.
        """)

    # ---------------- Feature Importance ----------------
    with col2:
        st.markdown("**Top Predictive Drivers**")

        top15 = feature_importance_df.head(15)

        fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=100)
        ax2.barh(top15["feature"], top15["pct_of_total"])
        ax2.invert_yaxis()
        ax2.set_xlabel("Importance", fontsize=9)
        ax2.set_ylabel("")
        ax2.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("""
        Both Random Forest and XGBoost independently surfaced similar drivers,  
        strengthening confidence in the stability of these findings.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # INTERPRETATION & CONCLUSION
    # =====================================================
    st.subheader("Interpretation & Conclusion")

    st.markdown("""
    • Structural strain is not random — it follows identifiable patterns  

    • Grade composition and socioeconomic concentration consistently  
      emerge as strong indicators  

    • Model agreement increases confidence in prioritization decisions  

    • This tool should support leadership — not replace local expertise  

    **Conclusion:**  
    Predictive modeling enables earlier identification, smarter prioritization,  
    and more proactive resource planning.
    """)

# =========================================================
# TAB 2 — SCHOOL DEEP DIVE
# =========================================================
with tab2:

    st.header("School Deep Dive")

    # -----------------------------------------------------
    # SELECTORS
    # -----------------------------------------------------
    school_list = sorted(df["SCH_NAME"].unique())
    selected_school = st.selectbox("Select School", school_list)

    years = sorted(df["SURVYEAR"].unique(), reverse=True)
    selected_year = st.selectbox("Select Year", ["All Years"] + years)

    school_all_years = df[df["SCH_NAME"] == selected_school].copy()

    # -----------------------------------------------------
    # HELPER FUNCTION — FORMAT TABLE
    # -----------------------------------------------------
    def format_display_table(dataframe):
        display_df = dataframe.copy()

        for col in display_df.columns:
            if display_df[col].dtype != "object":

                max_val = display_df[col].max()

                if max_val <= 1 and col != "high_strain":
                    display_df[col] = (display_df[col] * 100).round(2)

                elif max_val > 1:
                    display_df[col] = display_df[col].round(2)

        return display_df

    # -----------------------------------------------------
    # SINGLE YEAR MODE
    # -----------------------------------------------------
    if selected_year != "All Years":

        school_one_year = school_all_years[
            school_all_years["SURVYEAR"] == selected_year
        ].copy()

        st.subheader("School Snapshot")

        formatted_df = format_display_table(school_one_year)
        st.dataframe(formatted_df)

        # ---------------- Risk Assessment ----------------
        X_school = school_one_year[feature_list]

        predicted_prob = float(model.predict_proba(X_school)[:, 1][0])
        predicted_label = int(predicted_prob >= 0.5)

        actual_label = int(school_one_year["high_strain"].values[0])

        st.subheader("Risk Assessment")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Predicted Probability of Strain", f"{predicted_prob:.2%}")

        with colB:
            if predicted_label == 1:
                st.error("Predicted: High Strain")
            else:
                st.success("Predicted: Not High Strain")

        st.info(
            "Actual Label: High Strain"
            if actual_label == 1
            else "Actual Label: Not High Strain"
        )

        if predicted_label != actual_label:
            st.warning("⚠️ Model prediction does not match actual label.")

    # -----------------------------------------------------
    # ALL YEARS MODE
    # -----------------------------------------------------
    else:

        st.subheader("Multi-Year School Snapshot")
        st.dataframe(school_all_years.sort_values("SURVYEAR"))

        school_trend = school_all_years.sort_values("SURVYEAR")

        X_trend = school_trend[feature_list]
        y_probs = model.predict_proba(X_trend)[:, 1]

        years_arr = school_trend["SURVYEAR"].values

        # ===============================
        # MULTI-YEAR METRICS
        # ===============================
        latest_prob = y_probs[-1]

        peak_index = np.argmax(y_probs)
        peak_year = years_arr[peak_index]
        peak_prob = y_probs[peak_index]

        avg_prob = np.mean(y_probs)

        risk_growth = ((y_probs[-1] - y_probs[0]) / y_probs[0]) * 100

        st.subheader("Multi-Year Risk Assessment")

        colA, colB, colC, colD = st.columns(4)

        with colA:
            st.metric("Latest Year Risk", f"{latest_prob:.2%}")

        with colB:
            st.metric("Peak Risk Year", f"{peak_year} ({peak_prob:.2%})")

        with colC:
            st.metric("Average Multi-Year Risk", f"{avg_prob:.2%}")

        with colD:
            st.metric("Risk Growth Since Start", f"{risk_growth:.1f}%")

        st.info("Risk probabilities vary across observed years.")

        # ===============================
        # CHART + EXECUTIVE SUMMARY
        # ===============================
        col_chart, col_summary = st.columns([1.2, 1])

        # -------- Chart --------
        with col_chart:

            st.subheader("Predicted Risk Trajectory")

            fig_trend, ax_trend = plt.subplots(figsize=(4, 2.5), dpi=100)

            ax_trend.plot(years_arr, y_probs, marker="o")

            ax_trend.set_xlabel("Year", fontsize=8)
            ax_trend.set_ylabel("Probability of Strain", fontsize=8)

            ax_trend.tick_params(labelsize=8)

            plt.tight_layout()

            st.pyplot(fig_trend, use_container_width=False)

            st.caption(
                "Chart shows the model-estimated probability of structural strain over time."
            )

        # -------- Executive Summary --------
        with col_summary:

            st.subheader("Executive Trend Assessment")

            first_year = years_arr[0]
            last_year = years_arr[-1]

            first_prob = y_probs[0]
            last_prob = y_probs[-1]

            peak_index = np.argmax(y_probs)
            peak_year = years_arr[peak_index]
            peak_value = y_probs[peak_index]

            overall_change = last_prob - first_prob

            if overall_change > 0.05:
                trend_direction = "Overall worsening trend."
            elif overall_change < -0.05:
                trend_direction = "Overall improving trend."
            else:
                trend_direction = "Overall relatively stable trend."

            structural_status = (
                "Probability of strain remains elevated across observed years."
            )

            st.markdown(f"""
**This school is:**

• {structural_status}

• Peak risk occurred in **{peak_year}** at **{peak_value:.2%}**

• Risk changed from **{first_prob:.2%}** in {first_year}  
to **{last_prob:.2%}** in {last_year}

---

**Overall Pattern**

{trend_direction}

The most recent model estimate is **{last_prob:.2%}**, indicating continued structural strain risk.
""")

    # =====================================================
    # EXPANDER — HOW TO INTERPRET
    # =====================================================
    with st.expander("📘 How To Interpret These Numbers"):

        st.markdown("""
### Understanding the Scale of the Values

Binary Indicators  
• 1 = Yes  
• 0 = No  

Proportions  
• Values like 0.05 represent 5%

Percentages  
• Values like 23 represent 23%

Higher values indicate greater concentration of
that demographic or socioeconomic characteristic.
""")

    # =====================================================
    # EXPANDER — COLUMN DEFINITIONS
    # =====================================================
    with st.expander("📚 What Each Column Represents"):

        st.markdown("""
NCESSCH — NCES School ID  
SCH_NAME — School name  
SURVYEAR — Year of data

high_strain — 1 = school classified as strained

Income Distribution  
TH_10_15K — households earning $10–15k  
TH_15_25K — households earning $15–25k  
TH_50_75K — households earning $50–75k  
TH_75_100K — households earning $75–100k  
TH_100_150K — households earning $100–150k  
TH_150_200K — households earning $150–200k  
TH_200K_AND_ABOVE — households above $200k

Demographics  
prop_BL — % Black students  
prop_WH — % White students  
prop_HI — % Hispanic students  
prop_AS — % Asian students  
prop_AM — % American Indian students  
prop_TR — % two or more races

Economic Need  
frl_ratio — free/reduced lunch  
redl_ratio — reduced lunch only
""")