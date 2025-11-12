import streamlit as st
import pandas as pd
import pickle

# ==========================================
# 🌟 Page Configuration
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# ==========================================
# 🎨 Custom Styling
# ==========================================
st.markdown("""
    <style>
        /* General background */
        .main {
            background: linear-gradient(120deg, #f8f9fa 0%, #e9f7ef 100%);
            padding: 2rem;
            border-radius: 10px;
        }
        /* Title */
        h1 {
            text-align: center;
            color: #1e8449;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        h3 {
            color: #145a32;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f4f6f7;
        }
        /* Buttons */
        .stButton>button {
            background-color: #27ae60;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            font-size: 1rem;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1e8449;
            transform: scale(1.02);
        }
        /* Result boxes */
        .result-box {
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 20px;
        }
        .churn {
            background-color: #fdecea;
            color: #b03a2e;
        }
        .stay {
            background-color: #eafaf1;
            color: #1e8449;
        }
        /* Footer */
        .footer {
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 Load Model and Encoders
# ==========================================
try:
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Extract the trained model from dictionary
    loaded_model = model_data["model"]
    feature_names = model_data.get("features_names", None)

except Exception as e:
    st.error(f"⚠️ Error loading model or encoders: {e}")
    st.stop()

# ==========================================
# 🏷️ App Header
# ==========================================
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Use this intelligent dashboard to predict whether a telecom customer is likely to churn based on their profile and service details.")

# ==========================================
# 🧩 Sidebar Inputs
# ==========================================
st.sidebar.header("🧍‍♂️ Enter Customer Information")

col1, col2 = st.sidebar.columns(2)
gender = col1.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = col2.selectbox("Senior Citizen", [0, 1])
Partner = col1.selectbox("Partner", ["Yes", "No"])
Dependents = col2.selectbox("Dependents", ["Yes", "No"])

st.sidebar.markdown("---")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 1)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 29.85)
TotalCharges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 29.85)

# ==========================================
# 📦 Prepare Input Data
# ==========================================
input_data = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_data_df = pd.DataFrame([input_data])

# ==========================================
# 🔠 Encode Categorical Features
# ==========================================
try:
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])
except Exception as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# ==========================================
# 🔮 Predict Button
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([1, 2, 1])[1]

with center_col:
    if st.button("🔮 Predict Customer Churn"):
        try:
            prediction = loaded_model.predict(input_data_df)
            pred_prob = loaded_model.predict_proba(input_data_df)[0][1]

            if prediction[0] == 1:
                st.markdown(
                    f"""
                    <div class="result-box churn">
                        <h3>⚠️ Customer is likely to CHURN</h3>
                        <p><b>Churn Probability:</b> {pred_prob:.2%}</p>
                        <p>Recommendation: Consider offering discounts or loyalty programs to retain this customer.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                    <div class="result-box stay">
                        <h3>✅ Customer is likely to STAY</h3>
                        <p><b>Retention Probability:</b> {(1 - pred_prob):.2%}</p>
                        <p>Customer appears satisfied with services and billing model.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

