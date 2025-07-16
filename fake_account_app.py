import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load ML model
model = joblib.load("fake_account_model.pkl")

# Load images
header_img = Image.open("header.png")
fake_img = Image.open("fake.png")
real_img = Image.open("real.png")

# Page setup
st.set_page_config(page_title="Fake Account Detector", page_icon="🚩", layout="centered")

# Header
st.image(header_img, use_container_width=True)
st.title("🔍 Fake Account Detection on Social Media")
st.markdown("Check whether an account is **real or fake** using ML prediction.")

# Input form
st.subheader("📥 Enter Account Information")

account_age_days = st.number_input("📅 Account Age (in days)", min_value=0, value=30)
num_followers = st.number_input("👥 Number of Followers", min_value=0, value=100)
num_following = st.number_input("🔁 Number of Following", min_value=0, value=500)
num_posts = st.number_input("📝 Number of Posts", min_value=0, value=10)
profile_picture = st.selectbox("🖼️ Profile Picture Present?", ["Yes", "No"])
bio_filled = st.selectbox("✍️ Bio Filled?", ["Yes", "No"])

# Predict button
if st.button("🚀 Predict"):

    input_data = np.array([[
        account_age_days,
        num_followers,
        num_following,
        num_posts,
        1 if profile_picture == "Yes" else 0,
        1 if bio_filled == "Yes" else 0
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("🚩 The account is predicted as: **Fake Account**")
        st.image(fake_img, caption="Suspicious / Bot-like Activity", use_container_width=True)
    else:
        st.success("✅ The account is predicted as: **Real Account**")
        st.image(real_img, caption="Legitimate Account Detected", use_container_width=True)

    st.markdown("---")
