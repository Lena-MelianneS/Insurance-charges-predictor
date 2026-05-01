# Health Insurance Quote Generator

The Health Insurance generator provides a health insurance quote to the user based on their demographics attribute and provides a comparison with a state's average that is based on the Agency of Healthcare and Reasearch quality data. 

**Health Insurance Quote by Sarr Co.** [Click here](https://insurance-charges-predictor-mkmbkd8hy3d5skqhtb9mub.streamlit.app/)


## Table of Content
1. Project Overview
2. Project Structure 
3. Technology Used
3. Model Information
5. Results
6. Acknowledgments

### Project Overview

The project constructed aims to provide health insurance quotes to an user based on their personal attributes. The selected data set includes key elements such as Age, Sex, BMI, Number of Children, Smoking status and Region, all relevant factors in estimating insurance prices. The generator uses machine learning techniques to estimate prices based on the data set chosen and its correlation between variables. 

The analysis starts with data preprocessing to understand the correlations between variables, followed by an exploratary data analysis, feature engineering, development, then comparison of Linear Rgression, Polynomial Regression and Random Forest models to evaluate predictive performance. After evaluation, the bestperformance model is picked to estimate prices and deployed to the Streamlit app.
### Project Structure 
The project contains 2 main files;   and app.py. It essentially allows an user to get an insurance quote based on their demographics attrivute such as age, sex, bmi

**Health Insurance Analysis.ipynb** - The Jupyter Notebook contains the full machine learning pipeline including data preprocessing, including exploratory analysis and encoding categorical variables such as sex, smoker status and region. It additionally, stores Linear Regression, Polynomial Regression and Random Forest models training and evaluation. The best model is selected and saved based on performance.

**App.py** - The main Streamlit web application. Handles user inputs, loads the trained model, stores and displays the predicted insurance charges.

### Technology Used 
- Language : Python 3.11
- Environment : Jupyter Notebooks, Anaconda
- Data : CSV (Kaggle Insurance data, 1, 338 entries) and Medical Expenditure Panel Survey (MEPS) Insurance Component (IC) - Private Sector (State) from Agency for Healthcare Research and Quality
- **Libraries** : 
    - Pandas - Data Manipulation and cleaning
    - Numpy - Numerical computation
    - Matplotlib - Data Visualization
    - Seaborn - Statistical visualization 
    - Scipy - Statistical testing for Pearson correlation
    - Scikit-learn — Machine learning models and preprocessing
    - Joblib — Model serialization
- Web Application : Streamlit - interactive interface
- Source code storage : Github 
- Deployment : Streamlit Community Cloud; live app deployment
### Model Information
- **Dataset:** Kaggle csv Insurance Dataset
- **Models Tested:** Linear Regression, Polynomial Regression, Random Forest
- **Best Model:** Random Forest based on the highest R2
- **Attributes used:** Age, Sex, BMI, Number of Children, Smoking Status and Region
- **Target Variable:** Insurance charges
### Results
<img width="1166" height="1920" alt="16D7AB62-0B9A-4834-870A-D73243B297F8_1_201_a" src="https://github.com/user-attachments/assets/cd92aacf-14f4-43cf-8088-54399ae5391e" />

<img width="1155" height="1894" alt="7DB344C6-0309-4E45-9106-90970AE8657F_1_201_a" src="https://github.com/user-attachments/assets/8420e67b-7e9f-4b33-a707-886e6ff7697e" />

<img width="1169" height="1765" alt="39363EF2-E378-4C9E-8708-930515EA4FBD_1_201_a" src="https://github.com/user-attachments/assets/fbb7c19d-0d8a-4a17-8950-fcd39f2b8a68" />

### Acknowledgments
- Dataset: [Kaggle Insurance Dataset]((https://www.kaggle.com/datasets/lenameliannesarr/insurance))


