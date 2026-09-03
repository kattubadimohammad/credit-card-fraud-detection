import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_model():
    """Load the trusted trained model artifact."""
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure the model exists in the models directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Unable to load the model: {e}")
        return None


def predict(model, input_data):
    """Predict fraud probability for one transaction."""
    try:
        input_df = pd.DataFrame(input_data, columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        return prediction, probability
    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")
        return None, None


def plot_probability(probability):
    """Display the model's class probabilities."""
    labels = ["Legitimate", "Fraudulent"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, probability[0])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Confidence")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

    st.title("🚀 Credit Card Fraud Detection")
    st.write("Enter transaction details to predict whether the transaction is fraudulent.")

    st.markdown(
        """
        <style>
            div.stButton > button {
                padding: 10px 24px;
                margin-top: 20px;
            }
            div.stNumberInput > label { color: #333; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Transaction Features")
    st.caption("V1–V28 are anonymized PCA components from the original credit-card dataset.")

    input_data = []
    valid_range = (-10.0, 10.0)

    cols = st.columns(2)
    for i in range(1, 29):
        with cols[(i - 1) % 2]:
            value = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.4f",
                min_value=valid_range[0],
                max_value=valid_range[1],
                help=f"Anonymized Principal Component {i}",
            )
            input_data.append(value)

    amount = st.number_input(
        "💸 Transaction Amount",
        value=0.0,
        format="%.2f",
        min_value=0.0,
        help="Transaction amount in the original dataset units.",
    )
    input_data.append(amount)

    input_data = [input_data]

    predict_col, reset_col = st.columns(2)
    with predict_col:
        predict_clicked = st.button("🔎 Predict", use_container_width=True)
    with reset_col:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)

    if reset_clicked:
        st.rerun()

    if predict_clicked:
        model = load_model()
        if model is not None:
            with st.spinner("⏳ Predicting..."):
                result, probability = predict(model, input_data)

            if result is not None and probability is not None:
                fraud_probability = probability[0][1]
                legitimate_probability = probability[0][0]

                plot_probability(probability)

                if result[0] == 1:
                    st.error(
                        f"🚨 Fraudulent Transaction Detected! "
                        f"Confidence: {fraud_probability:.2%}"
                    )
                else:
                    st.success(
                        f"✅ Transaction is Legitimate. "
                        f"Confidence: {legitimate_probability:.2%}"
                    )

                with st.expander("📊 View Detailed Prediction Data"):
                    st.write(
                        f"Legitimate: {legitimate_probability:.2%} | "
                        f"Fraudulent: {fraud_probability:.2%}"
                    )
                    st.dataframe(
                        pd.DataFrame(input_data, columns=FEATURE_COLUMNS),
                        use_container_width=True,
                    )

                report_data = pd.DataFrame(
                    {
                        "Prediction": ["Fraudulent" if result[0] == 1 else "Legitimate"],
                        "Legitimate_Probability": [legitimate_probability],
                        "Fraudulent_Probability": [fraud_probability],
                    }
                )
                csv_data = report_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Report",
                    csv_data,
                    "fraud_detection_report.csv",
                    "text/csv",
                )


if __name__ == "__main__":
    main()
