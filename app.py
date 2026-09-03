from pathlib import Path

import pickle
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered",
)


def load_model():
    """Load the trusted trained model artifact."""
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure the model exists in the models directory.")
        return None
    except Exception as exc:
        st.error(f"⚠️ Unable to load the model: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def predict(model, input_data):
    """Predict fraud probability for one transaction."""
    try:
        input_df = pd.DataFrame(input_data, columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        return prediction, probability
    except Exception as exc:
        st.error(f"⚠️ Error while predicting: {exc}")
        return None, None


def render_confidence_bars(legitimate_probability, fraud_probability):
    """Render lightweight confidence bars without an external chart dependency."""
    st.markdown(
        "### 📊 Prediction Confidence"
    )

    st.write(f"**Legitimate:** {legitimate_probability:.2%}")
    st.progress(float(legitimate_probability))

    st.write(f"**Fraudulent:** {fraud_probability:.2%}")
    st.progress(float(fraud_probability))


def main():
    st.markdown(
        "<h1 style='text-align:center;'>💳 Credit Card Fraud Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#666;'>Analyze a transaction using a trained Random Forest model.</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "ℹ️ V1–V28 are anonymized PCA components from the credit-card dataset. "
        "For a meaningful prediction, enter a real or prepared feature vector from the same preprocessing pipeline."
    )

    st.subheader("🧾 Transaction Features")

    input_data = []
    valid_range = (-10.0, 10.0)
    cols = st.columns(2)

    for feature_number in range(1, 29):
        column_index = (feature_number - 1) % 2
        with cols[column_index]:
            value = st.number_input(
                f"V{feature_number}",
                value=0.0,
                format="%.4f",
                min_value=valid_range[0],
                max_value=valid_range[1],
                help=f"Anonymized PCA component V{feature_number}",
            )
            input_data.append(value)

    st.subheader("💸 Transaction Amount")
    amount = st.number_input(
        "Amount",
        value=0.0,
        format="%.2f",
        min_value=0.0,
        help="Transaction amount in the same original units used by the trained model.",
    )
    input_data.append(amount)

    st.divider()

    predict_col, reset_col = st.columns(2)

    with predict_col:
        predict_clicked = st.button(
            "🔎 Analyze Transaction",
            use_container_width=True,
            type="primary",
        )

    with reset_col:
        reset_clicked = st.button(
            "🔄 Reset",
            use_container_width=True,
        )

    if reset_clicked:
        st.rerun()

    if predict_clicked:
        model = get_model()

        if model is None:
            return

        with st.spinner("Analyzing transaction..."):
            result, probability = predict(model, [input_data])

        if result is None or probability is None:
            return

        legitimate_probability = float(probability[0][0])
        fraud_probability = float(probability[0][1])
        is_fraud = int(result[0]) == 1
        prediction_label = "Fraudulent" if is_fraud else "Legitimate"
        confidence = fraud_probability if is_fraud else legitimate_probability

        st.divider()
        st.subheader("🎯 Transaction Result")

        if is_fraud:
            st.error(
                f"🚨 Fraudulent Transaction Detected\n\n"
                f"**Confidence: {confidence:.2%}**"
            )
        else:
            st.success(
                f"✅ Transaction is Legitimate\n\n"
                f"**Confidence: {confidence:.2%}**"
            )

        render_confidence_bars(legitimate_probability, fraud_probability)

        with st.expander("📋 View Transaction Details"):
            details = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
            st.dataframe(details, use_container_width=True)

        report_data = pd.DataFrame(
            {
                "Prediction": [prediction_label],
                "Confidence": [confidence],
                "Legitimate_Probability": [legitimate_probability],
                "Fraudulent_Probability": [fraud_probability],
            }
        )

        st.download_button(
            "📥 Download Prediction Report",
            report_data.to_csv(index=False).encode("utf-8"),
            "fraud_detection_report.csv",
            "text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
