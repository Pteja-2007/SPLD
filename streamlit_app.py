# streamlit_app.py

import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_extraction.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit App Interface
st.title("📧 Spam Email Detection")
st.write("Enter your email message below and click **Predict** to check if it's Spam or Ham.")

# Input Text Area
user_input = st.text_area("✉️ Email Message", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform input
        input_features = vectorizer.transform([user_input])
        prediction = model.predict(input_features)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This is a **Ham** (not spam) message.")
        else:
            st.error("🚫 This is a **Spam** message.")
