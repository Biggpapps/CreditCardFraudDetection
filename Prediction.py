import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Background setup
def set_blurred_dark_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)), url({url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_blurred_dark_background("https://aihubprojects.com/wp-content/uploads/2020/01/ccfd-scaled.jpeg")

# Load model
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Tabs
tabs = st.tabs([
    "🏠 Home",
    "📊 EDA and Visualizations",
    "🔍 Fraud Prediction Model",
    "🤖 Credit Fraud Chatbot",
    "❓ Fraud FAQs"
])

# 🏠 Home Tab
with tabs[0]:
    st.title("🏠 Welcome to the Credit Card Fraud Detection App")
    st.markdown("""
    This project uses machine learning to predict whether a credit card application or transaction is likely fraudulent.
    The model was trained using 106 anonymized features from a Kaggle dataset.

    💡 **What you can do here:**
    - Explore visual trends via EDA screenshots
    - Predict fraud using smart/manual inputs or CSV
    - Ask a chatbot about credit fraud
    - Read FAQs asked by real fraud victims
    """)

# 📊 EDA and Visualizations Tab
with tabs[1]:
    st.title("📊 EDA and Visualizations")
    st.markdown("""
    Below is a snapshot of our exploratory data analysis from Google Colab.
    For a full interactive experience, visit the [Colab Notebook](https://colab.research.google.com/drive/1yogRmw0EFoETKz3TcXok-Wz2BjaYIT2S).
    """)
    st.image("C:/Users/HP/Pictures/Screenshots/Screenshot 2025-04-24 114112.png", caption="EDA Overview", use_container_width=True)
# 🔍 Prediction Tab
with tabs[2]:
    st.title("🔍 Credit Card Fraud Prediction")
    input_mode = st.radio("Choose input method:", ["🧠 Smart Input", "📂 Manual Entry", "📁 Upload CSV"])

    user_input = []
    submitted = False

    if input_mode == "🧠 Smart Input":
        st.subheader("🧠 Answer a few simple questions")
        amount = st.number_input("Transaction Amount", min_value=0.0)
        hour = st.slider("Hour of Transaction", 0, 23)
        region_match = st.radio("Transaction in your usual region?", ["Yes", "No"])
        frequent_today = st.radio("High usage today?", ["Yes", "No"])
        known_merchant = st.radio("Merchant well known?", ["Yes", "No"])
        name_contract = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        gender = st.radio("Gender", ["Male", "Female"])
        own_car = st.radio("Do you own a car?", ["Yes", "No"])
        own_realty = st.radio("Do you own real estate?", ["Yes", "No"])
        children = st.slider("Number of children", 0, 10)
        income = st.number_input("Annual Income", min_value=0.0)
        credit_amt = st.number_input("Credit Amount", min_value=0.0)
        annuity = st.number_input("Annuity Amount", min_value=0.0)

        mapped = [0.0] * 106
        mapped[0] = amount
        mapped[1] = hour
        mapped[2] = 1 if region_match == "Yes" else 0
        mapped[3] = 1 if frequent_today == "Yes" else 0
        mapped[4] = 1 if known_merchant == "Yes" else 0
        mapped[5] = 0 if name_contract == "Cash loans" else 1
        mapped[6] = 0 if gender == "Male" else 1
        mapped[7] = 1 if own_car == "Yes" else 0
        mapped[8] = 1 if own_realty == "Yes" else 0
        mapped[9] = children
        mapped[10] = income
        mapped[11] = credit_amt
        mapped[12] = annuity

        user_input = mapped
        submitted = st.button("🔍 Predict")

    elif input_mode == "📂 Manual Entry":
        st.subheader("📂 Enter all 106 Features")
        with st.form("manual_106"):
            for i in range(1, 107):
                val = st.number_input(f"Feature F{i}", value=0.0, step=0.01)
                user_input.append(val)
            submitted = st.form_submit_button("🔍 Predict")

    elif input_mode == "📁 Upload CSV":
        st.subheader("📁 Upload CSV File with F1 to F106")
        file = st.file_uploader("Upload a CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            if df.shape[1] != 106:
                st.error("CSV must have exactly 106 columns named F1 to F106.")
            else:
                st.write("📄 Uploaded Data Preview:")
                st.dataframe(df.head())
                if st.button("🔍 Predict"):
                    df["Prediction"] = model.predict(df)
                    df["Fraud Probability"] = model.predict_proba(df)[:, 1]
                    st.write(df)

    if submitted and len(user_input) == 106:
        arr = np.array([user_input])
        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0][1]

        st.subheader("🧾 Prediction Result")
        if pred == 1:
            st.error(f"🚨 **FRAUD** detected! (Probability: {prob:.2%})")
        else:
            st.success(f"✅ **LEGITIMATE** transaction. (Fraud Probability: {prob:.2%})")

# 🤖 Chatbot Tab
with tabs[3]:
    st.title("🤖 Credit Card Fraud Assistant")
    user_input = st.text_input("Ask anything about credit card fraud:")
    if user_input:
        if "fraud" in user_input.lower():
            st.info("Fraud is unauthorized use of your credit card. It can occur online or offline.")
        elif "prevent" in user_input.lower():
            st.info("Use secure passwords, monitor statements, and enable SMS alerts.")
        elif "machine learning" in user_input.lower() or "ml" in user_input.lower():
            st.info("ML helps identify unusual patterns and flag potentially fraudulent transactions.")
        else:
            st.info("Try asking about fraud, prevention, or how the model works!")

# ❓ FAQ Tab
with tabs[4]:
    st.title("❓ Common Questions About Fraud")
    faq_dict = {
        "1. What is credit card fraud?": "It's unauthorized use of a credit card to obtain funds or purchase items.",
        "2. How do I know if I’ve been a victim of fraud?": "Unexpected charges on your card or alerts from your bank.",
        "3. Can I get my money back after fraud?": "Yes, report it quickly and your bank can often reverse the transaction.",
        "4. What’s the safest way to shop online?": "Use secure sites, avoid public Wi-Fi, and enable card alerts.",
        "5. How does your model detect fraud?": "By learning patterns from historical data using 106 anonymized features.",
        "6. What features are used in prediction?": "Behavioral and transactional features, anonymized for privacy.",
        "7. Are ML models always accurate?": "No, they reduce risk but cannot catch 100% of fraud cases.",
        "8. What should I do immediately after fraud?": "Block the card, notify the bank, and file a fraud report.",
        "9. Can fraud happen even if my card is with me?": "Yes, if your card details are leaked or cloned.",
        "10. Is this model used in real banks?": "Similar models are used by financial institutions to assist analysts."
    }
    for q, a in faq_dict.items():
        st.markdown(f"**{q}**\n\n{a}\n")
