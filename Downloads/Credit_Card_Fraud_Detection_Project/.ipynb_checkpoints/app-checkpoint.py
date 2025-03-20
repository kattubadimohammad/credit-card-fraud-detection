import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

# Load the trained model
def load_model():
    try:
        with open('models/random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'random_forest_model.pkl' is in the 'models' directory.")
        return None

# Predict on user input
def predict(model, input_data):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        return prediction, probability
    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")
        return None, None

# Visualization with improved chart size and title
def plot_probability(probability):
    labels = ['Legitimate', 'Fraudulent']
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, probability[0], color=['green', 'red'])
    ax.set_ylabel('Confidence (%)')
    ax.set_ylim([0, 1])
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

def main():
    st.title('🚀 Credit Card Fraud Detection')
    st.write('Enter transaction details to predict if it is fraudulent or not.')

    # UI Styling
    st.markdown(
        """
        <style>
            div.stButton > button {
                background-color: #4CAF50; color: white; padding: 10px 24px; margin-top: 20px;
            }
            div.stNumberInput > label { color: #333; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # User inputs for the 28 principal components and amount
    input_data = []
    valid_range = (-10.0, 10.0)

    cols = st.columns(2)
    for i in range(1, 29):
        with cols[i % 2]:
            value = st.number_input(
                f'V{i}', 
                value=0.0, 
                format="%.4f", 
                min_value=float(valid_range[0]), 
                max_value=float(valid_range[1]), 
                help=f"Principal Component {i}"
            )
            input_data.append(value)

    amount = st.number_input('💸 Transaction Amount', value=0.0, format="%.2f", min_value=0.0)
    input_data.append(amount)

    input_df = pd.DataFrame([input_data])

    if st.button('🔎 Predict'):
        model = load_model()
        if model:
            with st.spinner('⏳ Predicting... Please wait'):
                time.sleep(2)
                result, probability = predict(model, input_df)

            if result is not None:
                plot_probability(probability)
                if result[0] == 1:
                    st.error(f'🚨 Fraudulent Transaction Detected! Confidence: {probability[0][1]*100:.2f}%')
                else:
                    st.success(f'✅ Transaction is Legitimate. Confidence: {probability[0][0]*100:.2f}%')
                
                # Detailed view with expandable option
                with st.expander("📊 View Detailed Prediction Data"):
                    st.write(f"Confidence Scores: Legitimate: {probability[0][0]*100:.2f}%, Fraudulent: {probability[0][1]*100:.2f}%")
                    st.write(input_df)

                # Download Report
                report_data = pd.DataFrame({'Prediction': ['Fraudulent' if result[0] == 1 else 'Legitimate'], 'Confidence': [max(probability[0])*100]})
                csv_data = report_data.to_csv(index=False).encode('utf-8')
                st.download_button('📥 Download Report', csv_data, 'report.csv', 'text/csv')

    if st.button('🔄 Reset'):
        st.experimental_rerun()

if __name__ == '__main__':
    main()