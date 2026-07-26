# Install Streamlit
import logging
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

# Quiet noisy warnings from older pickled models.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Make old sklearn pickle artifacts load correctly with current scikit-learn.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import _loss  # noqa: F401


# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    # Note: These filenames must match the ones used when saving the model and encoder.
    # Based on the previous cells, these should be 'churn_prediction_model.pkl' and 'one_hot_encoder.pkl'
    with open(ROOT / "churn_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ROOT / "one_hot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def build_input_dataframe(model, encoder, *, total_sessions, gross_session_length, active_days, active_quarters,
                          avg_sessions_per_quarter, avg_session_length_per_day, age, tech_comfort_score,
                          income_level, education, device_type):
    raw = pd.DataFrame([
        {
            "INCOME_LEVEL": income_level,
            "EDUCATION": education,
            "DEVICE_TYPE": device_type,
        }
    ])

    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    numeric_df = pd.DataFrame([
        {
            "TOTAL_SESSIONS": total_sessions,
            "GROSS_SESSION_LENGTH": gross_session_length,
            "ACTIVE_DAYS": active_days,
            "ACTIVE_QUARTERS": active_quarters,
            "AVG_SESSIONS_PER_QUARTER": avg_sessions_per_quarter,
            "AVG_SESSION_LENGTH_PER_DAY": avg_session_length_per_day,
            "AGE": age,
            "TECH_COMFORT_SCORE": tech_comfort_score,
        }
    ])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    input_df = input_df.reindex(columns=model.feature_names_in_)
    return input_df


def prepare_model_features(customer_data, model, encoder):
    required_numeric = [
        "TOTAL_SESSIONS",
        "GROSS_SESSION_LENGTH",
        "ACTIVE_DAYS",
        "ACTIVE_QUARTERS",
        "AVG_SESSIONS_PER_QUARTER",
        "AVG_SESSION_LENGTH_PER_DAY",
        "AGE",
        "TECH_COMFORT_SCORE",
    ]
    required_categorical = ["INCOME_LEVEL", "EDUCATION", "DEVICE_TYPE"]

    missing_numeric = [column for column in required_numeric if column not in customer_data.columns]
    missing_categorical = [column for column in required_categorical if column not in customer_data.columns]
    if missing_numeric or missing_categorical:
        raise ValueError(
            "Customer data is missing required columns: "
            f"numeric={missing_numeric}, categorical={missing_categorical}"
        )

    raw = customer_data[required_categorical].copy()
    for index, column in enumerate(required_categorical):
        valid_values = list(encoder.categories_[index])
        fallback_value = "Other" if "Other" in valid_values else valid_values[0]
        raw[column] = raw[column].apply(
            lambda value: value if pd.notna(value) and value in valid_values else fallback_value
        )

    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    numeric_df = customer_data[required_numeric].copy()

    input_df = pd.concat([numeric_df.reset_index(drop=True), encoded_df], axis=1)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    return input_df


def score_customer_data(customer_data, model, encoder):
    scored = customer_data.copy()
    model_input = prepare_model_features(scored, model, encoder)
    churn_probabilities = model.predict_proba(model_input)[:, 0]
    scored["churn_probability"] = churn_probabilities
    scored["renewal_probability"] = 1 - churn_probabilities
    scored["cltr"] = scored["ARR_2023_DEC"] * scored["GROSS_RETENTION_RATE_2023"]
    return scored


def build_quartile_summary(scored):
    scored = scored.copy()
    ranked_probabilities = scored["churn_probability"].rank(method="first")
    scored["quartile"] = pd.qcut(ranked_probabilities, q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    summary = (
        scored.groupby("quartile", sort=False)
        .agg(
            customer_count=("CUSTOMER_ID", "size"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_cltr=("cltr", "mean"),
        )
        .reset_index()
        .sort_values("quartile", key=lambda series: series.map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}))
    )
    return summary


def build_analysis_writeup(summary):
    highest_risk = summary.loc[summary["avg_churn_probability"].idxmax(), "quartile"]
    highest_value = summary.loc[summary["avg_cltr"].idxmax(), "quartile"]

    if highest_risk == highest_value:
        return (
            f"The {highest_risk} segment stands out as the strongest priority. It combines the highest average churn risk "
            "with the largest average CLTR, so retention spending there is most likely to protect both revenue and retention."
        )

    return (
        f"The {highest_risk} segment has the highest predicted churn risk, while the {highest_value} segment generates the highest "
        "average CLTR. The company should focus first on the {highest_risk} segment for proactive retention, and also protect "
        f"the {highest_value} segment because it contributes the most revenue per customer."
    )


def get_default_cohort_data():
    return pd.DataFrame([
        {
            "CUSTOMER_ID": "H1",
            "TOTAL_SESSIONS": 12,
            "GROSS_SESSION_LENGTH": 480,
            "ACTIVE_DAYS": 6,
            "ACTIVE_QUARTERS": 2,
            "AVG_SESSIONS_PER_QUARTER": 6,
            "AVG_SESSION_LENGTH_PER_DAY": 40,
            "AGE": 35,
            "TECH_COMFORT_SCORE": 7,
            "INCOME_LEVEL": "Low",
            "EDUCATION": "High School",
            "DEVICE_TYPE": "Mobile-only",
            "ARR_2023_DEC": 1000,
            "GROSS_RETENTION_RATE_2023": 0.82,
        },
        {
            "CUSTOMER_ID": "H2",
            "TOTAL_SESSIONS": 8,
            "GROSS_SESSION_LENGTH": 320,
            "ACTIVE_DAYS": 4,
            "ACTIVE_QUARTERS": 1,
            "AVG_SESSIONS_PER_QUARTER": 4,
            "AVG_SESSION_LENGTH_PER_DAY": 25,
            "AGE": 42,
            "TECH_COMFORT_SCORE": 4,
            "INCOME_LEVEL": "Medium",
            "EDUCATION": "Bachelor's",
            "DEVICE_TYPE": "Desktop",
            "ARR_2023_DEC": 1600,
            "GROSS_RETENTION_RATE_2023": 0.76,
        },
        {
            "CUSTOMER_ID": "H3",
            "TOTAL_SESSIONS": 20,
            "GROSS_SESSION_LENGTH": 700,
            "ACTIVE_DAYS": 10,
            "ACTIVE_QUARTERS": 4,
            "AVG_SESSIONS_PER_QUARTER": 8,
            "AVG_SESSION_LENGTH_PER_DAY": 50,
            "AGE": 29,
            "TECH_COMFORT_SCORE": 8,
            "INCOME_LEVEL": "High",
            "EDUCATION": "Graduate",
            "DEVICE_TYPE": "Mobile-only",
            "ARR_2023_DEC": 2200,
            "GROSS_RETENTION_RATE_2023": 0.88,
        },
        {
            "CUSTOMER_ID": "H4",
            "TOTAL_SESSIONS": 15,
            "GROSS_SESSION_LENGTH": 560,
            "ACTIVE_DAYS": 8,
            "ACTIVE_QUARTERS": 3,
            "AVG_SESSIONS_PER_QUARTER": 7,
            "AVG_SESSION_LENGTH_PER_DAY": 35,
            "AGE": 38,
            "TECH_COMFORT_SCORE": 6,
            "INCOME_LEVEL": "Medium",
            "EDUCATION": "High School",
            "DEVICE_TYPE": "Tablet",
            "ARR_2023_DEC": 1400,
            "GROSS_RETENTION_RATE_2023": 0.79,
        },
        {
            "CUSTOMER_ID": "H5",
            "TOTAL_SESSIONS": 18,
            "GROSS_SESSION_LENGTH": 620,
            "ACTIVE_DAYS": 7,
            "ACTIVE_QUARTERS": 2,
            "AVG_SESSIONS_PER_QUARTER": 5,
            "AVG_SESSION_LENGTH_PER_DAY": 30,
            "AGE": 31,
            "TECH_COMFORT_SCORE": 5,
            "INCOME_LEVEL": "Low",
            "EDUCATION": "Other",
            "DEVICE_TYPE": "Mobile-only",
            "ARR_2023_DEC": 1200,
            "GROSS_RETENTION_RATE_2023": 0.74,
        },
        {
            "CUSTOMER_ID": "H6",
            "TOTAL_SESSIONS": 9,
            "GROSS_SESSION_LENGTH": 350,
            "ACTIVE_DAYS": 5,
            "ACTIVE_QUARTERS": 2,
            "AVG_SESSIONS_PER_QUARTER": 4,
            "AVG_SESSION_LENGTH_PER_DAY": 22,
            "AGE": 46,
            "TECH_COMFORT_SCORE": 3,
            "INCOME_LEVEL": "High",
            "EDUCATION": "Post-Graduate",
            "DEVICE_TYPE": "Desktop-only",
            "ARR_2023_DEC": 1900,
            "GROSS_RETENTION_RATE_2023": 0.81,
        },
    ])


def get_artifacts():
    if "artifacts" not in st.session_state:
        with st.spinner("Loading the trained churn model..."):
            st.session_state.artifacts = load_artifacts()
    return st.session_state.artifacts


def run_app():
    model, encoder = get_artifacts()

    # ── UI ────────────────────────────────────────────────────────────────────────
    st.title("Healthy Meals Churn and CLTR Analysis")
    st.write(
        "This workflow uses the trained churn model to score Healthy Meals customers who were active as of 2025-01-01, "
        "splits them into equal-size quartiles by predicted churn probability, and compares average churn risk with average CLTR."
    )

    # Single-customer prediction section
    st.subheader("Single-customer prediction")
    st.write("Enter a customer profile to preview the predicted churn risk.")

    income_level_options = list(encoder.categories_[0])
    education_options = list(encoder.categories_[1])
    device_type_options = list(encoder.categories_[2])

    total_sessions = st.number_input("Total Sessions", min_value=0, value=12)
    gross_session_length = st.number_input("Gross Session Length", min_value=0, value=480)
    active_days = st.number_input("Active Days", min_value=0, value=6)
    active_quarters = st.number_input("Active Quarters", min_value=0, value=2)
    avg_sessions_per_quarter = st.number_input("Avg Sessions Per Quarter", min_value=0.0, value=6.0)
    avg_session_length_per_day = st.number_input("Avg Session Length Per Day", min_value=0.0, value=40.0)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income_level = st.radio("Income Level", income_level_options)
    education = st.radio("Education", education_options)
    device_type = st.radio("Device Type", device_type_options)
    tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)

    if st.button("Predict"):
        input_df = build_input_dataframe(
            model,
            encoder,
            total_sessions=total_sessions,
            gross_session_length=gross_session_length,
            active_days=active_days,
            active_quarters=active_quarters,
            avg_sessions_per_quarter=avg_sessions_per_quarter,
            avg_session_length_per_day=avg_session_length_per_day,
            age=age,
            tech_comfort_score=tech_comfort_score,
            income_level=income_level,
            education=education,
            device_type=device_type,
        )

        churn_probability = model.predict_proba(input_df)[0][0]
        renewal_probability = 1 - churn_probability
        risk = "Low" if churn_probability <= 0.4 else "Medium" if churn_probability <= 0.6 else "High"

        st.metric("Churn Probability", f"{churn_probability:.2f}")
        st.metric("Renewal Probability", f"{renewal_probability:.2f}")
        if risk == "High":
            st.error(f"Churn Risk: {risk}")
        elif risk == "Medium":
            st.warning(f"Churn Risk: {risk}")
        else:
            st.success(f"Churn Risk: {risk}")

    st.divider()
    st.subheader("Cohort quartile analysis")
    cohort_data = get_default_cohort_data()

    if st.button("Run cohort analysis"):
        try:
            scored_data = score_customer_data(cohort_data, model, encoder)
            summary = build_quartile_summary(scored_data)
            summary["avg_churn_probability"] = summary["avg_churn_probability"].round(3)
            summary["avg_cltr"] = summary["avg_cltr"].round(2)

            st.dataframe(summary, use_container_width=True)

            chart_data = summary.set_index("quartile")
            left_col, right_col = st.columns(2)
            with left_col:
                st.bar_chart(chart_data[["avg_churn_probability"]])
            with right_col:
                st.bar_chart(chart_data[["avg_cltr"]])

            st.write(build_analysis_writeup(summary))
        except ValueError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    if os.getenv("STREAMLIT_LAUNCHED_BY_PYTHON") == "1":
        run_app()
    else:
        os.environ["STREAMLIT_LAUNCHED_BY_PYTHON"] = "1"
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])