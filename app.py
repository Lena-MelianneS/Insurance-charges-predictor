import streamlit as st
import pandas as pd 
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
le_sex= joblib.load("label_encoderssex.pkl")
le_smoker=joblib.load("label_encoderssmoker.pkl")
le_region=joblib.load("label_encodersregion.pkl")
model=joblib.load("Best Model Random Forest_model.pkl")
df = pd.read_csv("insurance.csv")
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

state_monthly_premiums = {
    "Alabama": 500,
    "Alaska": 750,
    "Arizona": 450,
    "Arkansas": 600,
    "California": 500,
    "Colorado": 480,
    "Connecticut": 550,
    "Delaware": 520,
    "Florida": 550,
    "Georgia": 520,
    "Hawaii": 480,
    "Idaho": 537,
    "Illinois": 480,
    "Indiana": 500,
    "Iowa": 480,
    "Kansas": 500,
    "Kentucky": 520,
    "Louisiana": 530,
    "Maine": 550,
    "Maryland": 400,
    "Massachusetts": 450,
    "Michigan": 450,
    "Minnesota": 380,
    "Mississippi": 530,
    "Missouri": 500,
    "Montana": 550,
    "Nebraska": 520,
    "Nevada": 480,
    "New Hampshire": 370,
    "New Jersey": 520,
    "New Mexico": 480,
    "New York": 600,
    "North Carolina": 530,
    "North Dakota": 520,
    "Ohio": 460,
    "Oklahoma": 530,
    "Oregon": 480,
    "Pennsylvania": 470,
    "Rhode Island": 520,
    "South Carolina": 530,
    "South Dakota": 550,
    "Tennessee": 530,
    "Texas": 520,
    "Utah": 450,
    "Vermont": 1224,
    "Virginia": 460,
    "Washington": 480,
    "West Virginia": 700,
    "Wisconsin": 480,
    "Wyoming": 720,
}
# Source: Wealthvieu.com (April 2026) | CMS Government Data
# Monthly premiums for 40-year-old non-smoker, Silver plan, pre-subsidy


state_annual_premiums = {k: v * 12 for k, v in state_monthly_premiums.items()}

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

	# show prediction
       st.subheader("Your Estimated Cost")
       st.metric("Predicted Annual Charges", f"${prediction:,.2f}")
    
       # show region average comparison
       region = state_to_region[State]
       region_averages = df.groupby('region')['charges'].mean().round(2)
       avg = region_averages[region]
    
       st.subheader("How You Compare within our dataset")
       st.metric("Average Cost in Your Region", f"${avg:,.2f}")
    
       difference = prediction - avg
       if difference > 0:
            st.write(f"Your estimated cost is **${difference:,.2f} above** the {region} average")
       else:
            st.write(f"Your estimated cost is **${abs(difference):,.2f} below** the {region} average")
                                  
       state_avg_monthly = state_monthly_premiums[State]
       state_avg_annual = state_avg_monthly * 12

       st.subheader("Your Quote vs. Your State Average based on Wealthvieu 2026 data")
       col1, col2 = st.columns(2)
       col1.metric("Your Estimated Cost", f"${prediction:,.2f}/yr")
       col2.metric("State Average (Silver Plan)", f"${state_avg_annual:,.2f}/yr")

       difference = prediction - state_avg_annual
       if difference > 0:
         st.warning(f"Your estimate is ${difference:,.2f} above your state average")
       else:
         st.success(f"Your estimate is ${abs(difference):,.2f} below your state average")

       st.caption("Source: Wealthvieu.com (April 2026) | CMS Data | Silver plan, 40-year-old non-smoker, pre-subsidy")               