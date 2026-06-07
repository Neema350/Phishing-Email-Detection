import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Phishing Email Detection")

email = st.text_area("Paste Email Content")

if st.button("Check Email"):

    email_vector = vectorizer.transform([email])

    prediction = model.predict(email_vector)[0]

    if prediction.lower() == "phishing":
        st.error("⚠️ Phishing Email Detected")
    else:
        st.success("✅ Safe Email")