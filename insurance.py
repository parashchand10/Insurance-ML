import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and scaler
model = pickle.load(open('insurance_model.pkl', 'rb'))
scaler = pickle.load(open('insurance_scaler.pkl', 'rb'))

st.title("Medical Insurance Cost Predictor")
st.write("Enter the details below to estimate insurance charges.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Are you a smoker?", ["Yes", "No"])
region = st.selectbox("Region", ["Southeast", "Northeast", "Northwest", "Southwest"])

# Pre-processing input to match model features:
# Features: [age, children, is_smoker, region_southeast, bmi_category_obese, region_northeast, is_female]

if st.button("Predict Charges"):
    # 1. Transform categorical to numeric
    is_smoker = 1 if smoker == "Yes" else 0
    is_female = 1 if gender == "Female" else 0
    reg_se = 1 if region == "Southeast" else 0
    reg_ne = 1 if region == "Northeast" else 0
    bmi_obese = 1 if bmi >= 30 else 0
    
    # 2. Scale numerical features (Age and Children)
    # The scaler expects a 2D array of [age, children]
    scaled_nums = scaler.transform([[age, children]])
    scaled_age = scaled_nums[0][0]
    scaled_children = scaled_nums[0][1]
    
    # 3. Create Feature Array
    features = np.array([[
        scaled_age, 
        scaled_children, 
        is_smoker, 
        reg_se, 
        bmi_obese, 
        reg_ne, 
        is_female
    ]])
    
    # 4. Prediction
    prediction = model.predict(features)
    
    st.success(f"Estimated Insurance Charges: ${prediction[0]:,.2f}")
