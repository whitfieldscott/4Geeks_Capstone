import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

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
@st.cache_data
def load_data():
    return pd.read_csv("rf_production_dataset.csv")

@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    feature_list = joblib.load("rf_feature_list.pkl")
    return model, feature_list

df = load_data()
model, feature_list = load_model()

# ---------------------------------------------------------
# CREATE TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Project Overview", "School Deep Dive", "Model Evaluation"]
)

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

        importances = model.feature_importances_

        feature_importance = (
            pd.DataFrame({
                "feature": feature_list,
                "importance": importances
            })
            .sort_values("importance", ascending=False)
            .head(15)
        )

        fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=100)
        ax2.barh(feature_importance["feature"], feature_importance["importance"])
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
# TAB 2 — SCHOOL DEEP DIVE (POLISHED + EXPLAINABLE)
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
    # HELPER FUNCTION — FORMAT TABLE FOR DISPLAY
    # -----------------------------------------------------
    def format_display_table(dataframe):
        display_df = dataframe.copy()

        for col in display_df.columns:
            if display_df[col].dtype != "object":
                max_val = display_df[col].max()

                # Convert proportions (0–1 scale) into percentages
                if max_val <= 1 and col != "high_strain":
                    display_df[col] = (display_df[col] * 100).round(2)

                # Round regular percentage values
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
        predicted_label = int(predicted_prob >= 0.25)
        actual_label = int(school_one_year["high_strain"].values[0])

        st.subheader("Risk Assessment")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Predicted Risk Probability", f"{predicted_prob:.2%}")

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

        # Sort properly
        school_trend = school_all_years.sort_values("SURVYEAR")
        X_trend = school_trend[feature_list]
        y_probs = model.predict_proba(X_trend)[:, 1]

        years = school_trend["SURVYEAR"].values

        # ===============================
        # MULTI-YEAR RISK ASSESSMENT
        # ===============================

        latest_prob = y_probs[-1]
        peak_index = np.argmax(y_probs)
        peak_year = years[peak_index]
        peak_prob = y_probs[peak_index]
        avg_prob = np.mean(y_probs)

        threshold = 0.25
        years_above = np.sum(y_probs > threshold)

        st.subheader("Multi-Year Risk Assessment")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Latest Year Risk", f"{latest_prob:.2%}")

        with colB:
            st.metric("Peak Risk Year", f"{peak_year} ({peak_prob:.2%})")

        with colC:
            st.metric("Average Multi-Year Risk", f"{avg_prob:.2%}")

        if years_above == len(y_probs):
            st.warning("⚠️ Above threshold in every observed year.")
        elif years_above == 0:
            st.success("Below threshold in all observed years.")
        else:
            st.info("Mixed performance across years.")

        # ===============================
        # TRAJECTORY CHART (Controlled Size)
        # ===============================

        st.subheader("Predicted Risk Trajectory")

        fig_trend, ax_trend = plt.subplots(figsize=(4, 2.5), dpi=100)

        ax_trend.plot(years, y_probs, marker="o")
        ax_trend.axhline(threshold, linestyle="--")

        ax_trend.set_xlabel("Year", fontsize=8)
        ax_trend.set_ylabel("Predicted Risk", fontsize=8)
        ax_trend.tick_params(labelsize=8)

        plt.tight_layout()

        # IMPORTANT: prevent Streamlit stretching
        st.pyplot(fig_trend, use_container_width=False)

        st.markdown(
            """
            Shows predicted strain probability over time.
            Dashed line represents the 25% classification threshold.
            """
        )

        # ===============================
        # AUTOMATED TREND REPORT
        # ===============================

        first_year = years[0]
        last_year = years[-1]

        first_prob = y_probs[0]
        last_prob = y_probs[-1]

        peak_index = np.argmax(y_probs)
        peak_year = years[peak_index]
        peak_value = y_probs[peak_index]

        overall_change = last_prob - first_prob

        # Determine direction
        if overall_change > 0.05:
            trend_direction = "Overall worsening trend."
        elif overall_change < -0.05:
            trend_direction = "Overall improving trend."
        else:
            trend_direction = "Overall relatively stable trend."

        # Structural persistence
        if years_above == len(y_probs):
            structural_status = "Above threshold every observed year."
        else:
            structural_status = "Not consistently above threshold."

        st.markdown("### 📊 Executive Trend Assessment")

        st.markdown(f"""
        **This school is:**

        - {structural_status}
        - Peak risk occurred in **{peak_year}** at **{peak_value:.2%}**
        - Risk changed from **{first_prob:.2%}** in {first_year}  
        to **{last_prob:.2%}** in {last_year}

        ---

        **Overall Pattern:**

        {trend_direction}

        Even at **{last_prob:.2%}**, this school remains  
        **{last_prob/threshold:.1f}× above the 25% threshold**,  
        indicating continued structural strain risk.
        """)

    # =====================================================
    # EXPANDER 1 — HOW TO INTERPRET THE NUMBERS
    # =====================================================
    with st.expander("📘 How To Interpret These Numbers"):

        st.markdown("""
        ### Understanding the Scale of the Values

        This dataset includes different numeric types:

        #### 1️⃣ Binary Indicators
        • **1 = Yes / True**  
        • **0 = No / False**  
        Example: `high_strain`

        #### 2️⃣ Percentage Values (Already in %)
        Values like **2.2, 24.9, 15.2** represent percentages directly.  
        Example:  
        `24.9` = 24.9% of households fall in that income band.

        #### 3️⃣ Proportion Values (0–1 Scale)
        Values like **0.0506** represent proportions.  
        Example:  
        `0.0506` = 5.06% of students.

        ---

        Higher values indicate greater concentration of that
        demographic or economic characteristic.
        """)

    # =====================================================
    # EXPANDER 2 — WHAT EACH COLUMN MEANS
    # =====================================================
    with st.expander("📚 What Each Column Represents"):

        st.markdown("""
        ### Core Identifiers

        • **NCESSCH** – National Center for Education Statistics school ID  
        • **SCH_NAME** – School name  
        • **SURVYEAR** – School year  

        ### Strain Indicator

        • **high_strain** – 1 = School identified as overcrowded / strained  
          0 = Not identified as strained  

        ### Income Distribution (Household %)

        • **TH_10_15K** – % households earning $10k–$15k  
        • **TH_15_25K** – % earning $15k–$25k  
        • **TH_50_75K** – % earning $50k–$75k  
        • **TH_75_100K** – % earning $75k–$100k  
        • **TH_100_150K** – % earning $100k–$150k  
        • **TH_150_200K** – % earning $150k–$200k  
        • **TH_200K_AND_ABOVE** – % earning over $200k  

        ### Assistance Indicators

        • **TH_WITH_SNAP** – % households receiving SNAP benefits  
        • **TH_WITH_CASH_ASSIST** – % receiving cash assistance  

        ### Demographic Composition (Student %)

        • **prop_BL** – % Black students  
        • **prop_WH** – % White students  
        • **prop_HI** – % Hispanic students  
        • **prop_AS** – % Asian students  
        • **prop_AM** – % American Indian students  
        • **prop_TR** – % Two or more races  

        ### Economic Need

        • **frl_ratio** – % eligible for Free/Reduced Lunch  
        • **redl_ratio** – Reduced lunch only ratio  

        ---

        These indicators help evaluate socioeconomic concentration,
        demographic composition, and potential strain drivers.
        """)

# =========================================================
# TAB 3 — MODEL EVALUATION
# =========================================================
with tab3:

    st.header("Model Evaluation")

    from sklearn.metrics import (
        roc_curve,
        precision_recall_curve,
        roc_auc_score,
        average_precision_score,
    )

    X_all = df[feature_list]
    y_true = df["high_strain"]
    y_probs = model.predict_proba(X_all)[:, 1]

    # -------- ROC --------
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # -------- PR --------
    st.subheader("Precision-Recall Curve")

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # -------- Metrics --------
    colM1, colM2 = st.columns(2)
    with colM1:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    with colM2:
        st.metric("Average Precision", f"{avg_precision:.3f}")