import streamlit as st
import pandas as pd 
import numpy as np
import joblib

#load data 
df = pd.read_csv("insurance.csv")

#Open necessary pkl files
scaler = joblib.load("scaler.pkl")
le_sex= joblib.load("label_encoderssex.pkl")
le_smoker=joblib.load("label_encoderssmoker.pkl")
le_region=joblib.load("label_encodersregion.pkl")
model=joblib.load("Best Model Random Forest_model.pkl")

#Creating a dropdown list of States and assigning them to matching regions
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
# Average total employee contribution (in dollars) per enrolled employee for employee-plus-one coverage
# at private-sector establishments that offer health insurance in the US from  2022-2024, a 3-year average
average_state_annual = {
    "Alabama": 4459,
    "Alaska": 4723,
    "Arizona": 4709,
    "Arkansas": 4662,
    "California": 4448,
    "Colorado": 4631,
    "Connecticut": 4291,
    "Delaware": 4649,
	"District of Columbia":4608,
    "Florida": 5087,
    "Georgia": 4261,
    "Hawaii": 4009,
    "Idaho": 4322,
    "Illinois": 4477,
    "Indiana": 4251,
    "Iowa": 4253,
    "Kansas": 4284,
    "Kentucky": 4356,
    "Louisiana": 5515,
    "Maine": 4420,
    "Maryland": 4584,
    "Massachusetts": 4001,
    "Michigan": 4253,
    "Minnesota": 4363,
    "Mississippi": 4345,
    "Missouri": 4830,
    "Montana": 4475,
    "Nebraska": 4270,
    "Nevada": 3930,
    "New Hampshire": 4358,
    "New Jersey": 4365,
    "New Mexico": 4711,
    "New York": 4322,
    "North Carolina": 5268,
    "North Dakota": 4078,
    "Ohio": 4077,
    "Oklahoma": 4428,
    "Oregon": 3637,
    "Pennsylvania": 4191,
    "Rhode Island": 4192,
    "South Carolina":4486 ,
    "South Dakota": 5082,
    "Tennessee": 4540,
    "Texas": 4949,
    "Utah": 3908,
    "Vermont": 4939,
    "Virginia": 4506,
    "Washington": 3976,
    "West Virginia": 3901,
    "Wisconsin": 4104,
    "Wyoming": 4661,
}
# Source: MEPS-IC Data Tools | Agency for Healthcare Research and Quality

average_state_monthly  = {k: v / 12 for k, v in average_state_annual.items()}

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
       st.metric("Predicted Annual Charges", f"${prediction:,.2f}, thus ${prediction/12:,.2f} per month")
    
       # show region average comparison
       region = state_to_region[State]
       region_averages = df.groupby('region')['charges'].mean().round(2)
       avg = region_averages[region]
    
       st.subheader("Comparison to our dataset")
       st.metric("Average Cost in Your Region", f"${avg:,.2f}")
    
       difference = prediction - avg
       if difference > 0:
            st.write(f"Your estimated cost is **${difference:,.2f} above** the {region} average")
       else:
            st.write(f"Your estimated cost is **${abs(difference):,.2f} below** the {region} average")
                                  
       state_avg_monthly = average_state_annual[State]
       state_avg_annual = average_state_annual

       st.subheader("Your Quote vs. Your State Average ")
       col1, col2 = st.columns(2)
       col1.metric("Your Estimated Cost", f"${prediction:,.2f}/yr")
       col2.metric("State Average Based on 2022-2024 Data", f"${state_avg_annual:,.2f}/yr")

       difference = prediction - state_avg_annual
       if difference > 0:
         st.warning(f"Your estimate is ${difference:,.2f} above your state average")
       else:
         st.success(f"Your estimate is ${abs(difference):,.2f} below your state average")

       st.caption("Source: Medical Expenditure Panel Survey (MEPS) Insurance Component (IC) - Private Sector (State) | Agency for Healthcare Research and Quality")               
