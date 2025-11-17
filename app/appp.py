# ======================================================
# Credit Card Fraud Detection - Streamlit App (All 30 Features)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------
# Load the trained Random Forest model
# ------------------------------------------------------
model_path = "D:\\domain projects\\credit card\\Models\\random_forest.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# ------------------------------------------------------
# App Header
# ------------------------------------------------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Predict fraudulent transactions using a trained **Random Forest** model with all 30 features.")

# ------------------------------------------------------
# Tabs for different functionalities
# ------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔹 Single Prediction", "📁 Bulk Prediction (CSV)", "📊 Dashboard"])

# ======================================================
# TAB 1 - Single Prediction
# ======================================================
with tab1:
    st.subheader("Enter Transaction Details (All 30 Features)")

    st.markdown("Enter the values for **Time, V1–V28, and Amount** below:")

    # Layout for user-friendly input (split into columns)
    col1, col2, col3 = st.columns(3)

    # Collect inputs
    Time = col1.number_input("Time", value=0.0)
    Amount = col3.number_input("Amount", value=0.0)

    # Create dynamic inputs for V1–V28
    v_features = []
    for i in range(1, 29):
        if i <= 10:
            v_features.append(col1.number_input(f"V{i}", value=0.0))
        elif i <= 19:
            v_features.append(col2.number_input(f"V{i}", value=0.0))
        else:
            v_features.append(col3.number_input(f"V{i}", value=0.0))

    if st.button("🔍 Predict Fraud Status"):
        # Prepare input in correct shape (1, 30)
        input_data = np.array([[Time] + v_features + [Amount]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 Transaction is **Fraudulent!**")
        else:
            st.success("✅ Transaction is **Legitimate.**")

# ======================================================
# TAB 2 - Bulk Prediction (CSV Upload)
# ======================================================
with tab2:
    st.subheader("Upload CSV File for Fraud Detection (must have 30 feature columns)")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:", data.head())

        # Predict
        try:
            preds = model.predict(data)
            data["Prediction"] = preds
            st.write("### Prediction Results:")
            st.dataframe(data.head(10))

            fraud_count = np.sum(preds == 1)
            legit_count = np.sum(preds == 0)

            st.info(f"✅ Legitimate Transactions: {legit_count}")
            st.error(f"🚨 Fraudulent Transactions: {fraud_count}")

            # Download predictions
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error while predicting: {e}")

# ======================================================
# TAB 3 - Dashboard / Visualization
# ======================================================
with tab3:
    st.subheader("Fraud Detection Dashboard")

    uploaded_data = st.file_uploader("Upload a CSV file with predictions", type=["csv"], key="dashboard")
    if uploaded_data is not None:
        df = pd.read_csv(uploaded_data)

        if "Prediction" not in df.columns:
            st.warning("⚠️ The uploaded file doesn't contain 'Prediction' column. Please use the output CSV from Tab 2.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.write("### 🔹 Fraud vs Legitimate")
                fig, ax = plt.subplots()
                sns.countplot(x="Prediction", data=df, palette="coolwarm", ax=ax)
                ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
                st.pyplot(fig)

            with col2:
                st.write("### 🔹 Fraud Proportion (%)")
                fraud_count = df["Prediction"].value_counts(normalize=True) * 100
                fig2, ax2 = plt.subplots()
                ax2.pie(fraud_count, labels=["Legit", "Fraud"], autopct="%1.1f%%", colors=["#00cc66", "#ff3300"])
                st.pyplot(fig2)

            st.write("### 💰 Transaction Amount Distribution by Prediction")
            fig3, ax3 = plt.subplots()
            sns.histplot(data=df, x="Amount", hue="Prediction", bins=50, kde=True, palette="coolwarm", ax=ax3)
            st.pyplot(fig3)
