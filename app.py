import streamlit as st
import pandas as pd 
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
le_sex= joblib.load("label_encoderssex.pkl")
le_smoker=joblib.load("label_encoderssmoker.pkl")
le_region=joblib.load("label_encodersregion.pkl")
model=joblib.load("Best Model Random Forest_model.pkl")

state_to_region={
    "Connecticut": "northeast",
    "Maine": "northeast",
    "Massachusetts": "northeast",
    "New Hampshire": "northeast",
    "Rhode Island": "northeast",
    "Vermont": "northeast",
    "New Jersey": "northeast",
    "New York": "northeast",
    "Pennsylvania": "northeast",

    "Illinois": "northwest",
    "Indiana": "northwest",
    "Iowa": "northwest",
    "Kansas": "northwest",
    "Michigan": "northwest",
    "Minnesota": "northwest",
    "Missouri": "northwest",
    "Nebraska": "northwest",
    "North Dakota": "northwest",
    "Ohio": "northwest",
    "South Dakota": "northwest",
    "Wisconsin": "northwest",

    "Alabama": "southeast",
    "Arkansas": "southeast",
    "Delaware": "southeast",
    "Florida": "southeast",
    "Georgia": "southeast",
    "Kentucky": "southeast",
    "Louisiana": "southeast",
    "Maryland": "southeast",
    "Mississippi": "southeast",
    "North Carolina": "southeast",
    "South Carolina": "southeast",
    "Tennessee": "southeast",
    "Virginia": "southeast",
    "West Virginia": "southeast",
    "District of Columbia": "southeast",

    "Arizona": "southwest",
    "California": "southwest",
    "Colorado": "southwest",
    "Idaho": "southwest",
    "Montana": "southwest",
    "Nevada": "southwest",
    "New Mexico": "southwest",
    "Oklahoma": "southwest",
    "Oregon": "southwest",
    "Texas": "southwest",
    "Utah": "southwest",
    "Washington": "southwest",
    "Wyoming": "southwest",
    "Alaska": "southwest",
    "Hawaii": "southwest"
}

st.set_page_config(page_title = "Insurance Claim Predictor", layout="centered")
st.title("Health insurance Payment Prediction App by Sarr company")
st.write("Enter the details below to estimate you insurance payment amount:")

with st.form("input_form"):
    col1, col2 =st.columns(2)
    with col1:
        age = st.number_input("age", min_value =0, max_value=100, value=30)
        bmi = st.number_input("bmi",min_value =10.0, max_value = 40.0, value =24.0)
        children=st.number_input("Number of Children", min_value=0,max_value=20,value=0)
    with col2:
        smoker=st.selectbox("smoker",options=le_smoker.classes_)
        State=st.selectbox("Select your State:", sorted(state_to_region.keys()))
        region = state_to_region[State]
        st.write("Mapped Region:", region)
        sex= st.selectbox("sex", options =le_sex.classes_)
                           
    submitted = st.form_submit_button("Estimate Payment")

if submitted :
       input_data= pd.DataFrame({
           "age":[age],
           "sex":[sex],
           "bmi":[bmi],
           "Number of Children":[children],
           "smoker": [smoker],
           "region":[region],
           
       })
                           
       input_data["sex"]=le_sex.transform(input_data["sex"])
       input_data["smoker"]=le_smoker.transform(input_data["smoker"])
       input_data["region"]=le_region.transform(input_data["region"])
            
       num_cols=["age","bmi","Number of Children"]
       input_data[num_cols]=scaler.transform(input_data[num_cols])
       prediction = model.predict(input_data)[0]
       st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
                           
                           