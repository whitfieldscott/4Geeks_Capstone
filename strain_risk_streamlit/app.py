import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import requests
import tempfile
import shap
from openai import OpenAI
from sklearn.metrics import roc_curve, auc

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
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

df = load_data()

# ---------------------------------------------------------
# FEATURE LABELS (Human-readable names for SHAP plots)
# ---------------------------------------------------------

feature_labels = {
    "TH_WITH_SNAP": "Households Receiving SNAP",
    "TH_LT_10K": "Households earning < $10K",
    "TH_10_15K": "Households earning $10–15K",
    "TH_15_25K": "Households earning $15–25K",
    "TH_35_50K": "Households earning $35–50K",
    "TH_50_75K": "Households earning $50–75K",
    "TH_75_100K": "Households earning $75–100K",
    "TH_100_150K": "Households earning $100–150K",
    "TH_150_200K": "Households earning $150–200K",
    "TH_200K_AND_ABOVE": "Households earning $200K+",

    "prop_BL": "% Black Students",
    "prop_WH": "% White Students",
    "prop_HI": "% Hispanic Students",
    "prop_AS": "% Asian Students",
    "prop_AM": "% American Indian Students",
    "prop_TR": "% Two or More Races",

    "frl_ratio": "Free/Reduced Lunch Ratio",
    "redl_ratio": "Reduced Lunch Ratio Only",

    "BPL_ALL": "Population Below Poverty Level",
    "locale_category_Rural": "Rural School Location"
}

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
# LOAD SHAP EXPLAINER
# ---------------------------------------------------------
@st.cache_resource
def load_shap_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return explainer

explainer = load_shap_explainer(model)

# ---------------------------------------------------------
# OPENAI SUMMARY FUNCTION
# ---------------------------------------------------------

@st.cache_data
def generate_ai_summary(predicted_prob, year, school_name, shap_features):

    prompt = f"""
You are an education data analyst.

A machine learning model predicted a structural strain risk for a school.

School: {school_name}
Year: {year}
Predicted strain probability: {predicted_prob:.2%}

Top contributing factors:
{shap_features}

Write a short executive-style interpretation explaining what this means and what administrators should consider.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

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

    st.markdown("### Research Question")

    st.markdown(
    """
    <div style="max-width:700px;">
    Can school and community characteristics help predict which schools
    experience higher student-teacher ratios?
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Project Framing")

    st.markdown(
    """
    <div style="max-width:700px;">
    We used machine learning to analyze national school and socioeconomic
    data to identify patterns associated with higher student-teacher ratios.
    </div>
    """,
    unsafe_allow_html=True
    )

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

    X_all = df[feature_list]
    all_probs = model.predict_proba(X_all)[:, 1]

    col1, col2 = st.columns(2)

# ---------------- ROC-AUC PERFORMANCE ----------------
with col1:
    st.markdown("**Model Performance (ROC-AUC Curve)**")

     # actual values
    y_true = df["high_strain"]

    # predicted probabilities
    y_scores = model.predict_proba(df[feature_list])[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5,3), dpi=100)

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")

    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.tick_params(labelsize=9)

    ax.legend(loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"""
    The ROC curve evaluates how well the model distinguishes between schools
    experiencing structural strain and those that are not.

    The model achieved an **AUC score of {roc_auc:.3f}**, indicating strong
    predictive capability. An AUC close to 1.0 means the model can reliably
    separate higher-risk schools from lower-risk schools across a range of
    classification thresholds.
    """)


# ---------------- PRECISION vs RECALL ----------------
with col2:
    st.markdown("**XGBoost: Precision vs Recall**")

    from sklearn.metrics import precision_score, recall_score

    # predictions
    y_pred = (model.predict_proba(df[feature_list])[:,1] > 0.5).astype(int)

    precision = precision_score(df["high_strain"], y_pred)
    recall = recall_score(df["high_strain"], y_pred)

    fig2, ax2 = plt.subplots(figsize=(5,3), dpi=100)

    metrics = ["Precision", "Recall"]
    values = [precision, recall]

    bars = ax2.bar(metrics, values)

    ax2.set_ylim(0,1)
    ax2.set_ylabel("Score", fontsize=9)
    ax2.tick_params(labelsize=9)

    # add percentages above bars
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.02,
            f"{val*100:.1f}%",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("""
    Precision measures how often the model is correct when it predicts a school
    is experiencing structural strain. Recall measures how effectively the model
    identifies schools that truly face elevated strain risk.

    The model prioritizes **recall**, meaning it is more focused on identifying
    potentially strained schools rather than missing them, which is appropriate
    for early-warning monitoring systems used by education administrators.
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
    # SCHOOL SELECTOR
    # -----------------------------------------------------
    school_lookup = (
        df[["SCH_NAME", "NCESSCH"]]
        .drop_duplicates()
        .sort_values("SCH_NAME")
        .reset_index(drop=True)
    )

    school_lookup["display_name"] = (
        school_lookup["SCH_NAME"] + " (ID: " + school_lookup["NCESSCH"].astype(str) + ")"
    )

    selected_display = st.selectbox(
        "Select School",
        school_lookup["display_name"]
    )

    selected_nces = school_lookup.loc[
        school_lookup["display_name"] == selected_display,
        "NCESSCH"
    ].iloc[0]

    school_all_years = df[df["NCESSCH"] == selected_nces].copy()


    # -----------------------------------------------------
    # YEAR SELECTOR
    # -----------------------------------------------------
    years = sorted(school_all_years["SURVYEAR"].unique(), reverse=True)

    selected_year = st.selectbox(
        "Select Year",
        ["All Years"] + years
    )

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

                
        # -----------------------------------------------------
        # SHAP EXPLANATION
        # -----------------------------------------------------
        st.subheader("Key Drivers of This Prediction")

        shap_values = explainer(X_school)
        
        shap_df = pd.DataFrame({
            "feature": feature_list,
            "impact": shap_values.values[0]
        })

        top_features = (
            shap_df
            .assign(abs_impact=lambda x: np.abs(x["impact"]))
            .sort_values("abs_impact", ascending=False)
            .head(10)
        )

        top_feature_list = top_features["feature"].tolist()

        fig_shap, ax_shap = plt.subplots(figsize=(4,2.5), dpi=100)

        # Convert feature names to readable labels
        display_features = [feature_labels.get(f, f) for f in top_features["feature"]]

        ax_shap.barh(display_features, top_features["impact"])

        ax_shap.axvline(0)
        ax_shap.invert_yaxis()
        ax_shap.set_xlabel("Impact on Prediction", fontsize=8)
        ax_shap.tick_params(labelsize=8)

        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=False)

        ai_summary = generate_ai_summary(
            predicted_prob,
            selected_year,
            school_one_year["SCH_NAME"].values[0],
            top_feature_list
        )

        st.subheader("AI Interpretation")

        st.info(ai_summary)

        # -----------------------------------------------------
        # ASK AI ABOUT THIS SCHOOL
        # -----------------------------------------------------

        st.subheader("AI Policy Assistant")

        user_question = st.text_input(
            "Ask a question about this school's strain risk:"
        )

        if st.button("Ask AI") and user_question:

            prompt = f"""
            You are an education policy analyst.

            School: {school_one_year["SCH_NAME"].values[0]}
            Year: {selected_year}

            Predicted strain probability: {predicted_prob:.2%}

            Key contributing factors:
            {top_feature_list}

            User question:
            {user_question}

            Provide a concise explanation and possible policy or administrative actions.
            """

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content

            st.markdown("### AI Response")
            st.write(answer)

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


        # -----------------------------------------------------
        # RISK INTERPRETATION LEVEL
        # -----------------------------------------------------

        with col_summary:

                # interpretation logic
            if last_prob < 0.10:
                interpretation = "low predicted strain risk"
            elif last_prob < 0.25:
                interpretation = "mild strain risk"
            elif last_prob < 0.50:
                interpretation = "moderate strain risk"
            elif last_prob < 0.75:
                interpretation = "high strain risk"
            else:
                interpretation = "severe strain risk"

            st.markdown(f"""
            **This school is:**

            • {structural_status}

            • Peak risk occurred in **{peak_year}** at **{peak_value:.2%}**

            • Risk changed from **{first_prob:.2%}** in {first_year}
            to **{last_prob:.2%}** in {last_year}

            **Overall Pattern**

            {trend_direction}

            The most recent model estimate is **{last_prob:.2%}**, indicating **{interpretation}**.
            """)

            # -----------------------------------------------------
            # AI POLICY ASSISTANT (MULTI-YEAR ANALYSIS)
            # -----------------------------------------------------

            st.subheader("AI Policy Assistant (Multi-Year Analysis)")

            multi_year_question = st.text_input(
                "Ask a question about this school's long-term strain trends:"
            )

            if st.button("Ask AI About Trends") and multi_year_question:

                trend_prompt = f"""
                You are an education policy analyst.

                School: {school_all_years.SCH_NAME.iloc[0]}

                Observed years: {years_arr}

                Latest strain probability: {latest_prob:.2%}
                Average strain probability: {avg_prob:.2%}

                Peak risk year: {peak_year} ({peak_value:.2%})

                Overall trend:
                {trend_direction}

                User question:
                {multi_year_question}

                Provide a concise explanation and possible administrative or policy actions.
                """

                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": trend_prompt}]
                )

                trend_answer = response.choices[0].message.content

                st.markdown("### AI Trend Analysis")
                st.write(trend_answer)

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