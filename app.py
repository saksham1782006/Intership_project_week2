import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Load model from Hugging Face Hub
# -----------------------------
repo_id = "saksham1122211/Intership_project_model"

model_path = hf_hub_download(repo_id=repo_id, filename="rf_model.pkl")
model = joblib.load(model_path)

# -----------------------------
# Load local scaler & feature list
# -----------------------------
scaler = joblib.load("scaler.pkl")              # keep this file in repo
feature_list = joblib.load("feature_columns.pkl")  # keep this file in repo

if not isinstance(feature_list, list):
    feature_list = list(feature_list)

# -----------------------------
# UI
# -----------------------------
st.title("🏢 Building Energy Prediction App")
st.write("Enter building details below to predict **Site EUI**")

state = st.selectbox("Select State Factor", [1, 2, 3, 4, 5, 6, 7, 8])
building_class = st.selectbox("Building Class", ["Residential", "Commercial"])
facility_type = st.selectbox("Facility Type", [
    "Grocery_store_or_food_market",
    "Warehouse_Distribution_or_Shipping_center",
    "Retail_Enclosed_mall",
    "2to4_Unit_Building",
    "Office",
    "Commercial_Other"
])

year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
floor_area = st.number_input("Floor Area (sqft)", min_value=100, max_value=1000000, value=1000)
energy_star = st.number_input("Energy Star Rating", min_value=0, max_value=100, value=50)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Site EUI"):
    input_df = pd.DataFrame([{
        "State_Factor": state,
        "building_class": building_class,
        "facility_type": facility_type,
        "year_built": year_built,
        "floor_area": floor_area,
        "energy_star_rating": energy_star
    }])

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=feature_list, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"✅ Predicted Site EUI: {prediction:.2f}")

