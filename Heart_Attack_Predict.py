#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# Load the model
heart_disease_model = pickle.load(open('C:/Users/conte/heart_disease_model.sav','rb'))

# Define the prediction function
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    return prediction[0][0]

# Create a Streamlit app
st.title("Heart Disease Predictor")

# Add an input form for user inputs
age = st.number_input("Age")
sex = st.number_input("Sex (1 for male, 0 for female)")
cp = st.number_input("Chest Pain Type (0 to 3)")
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Serum Cholestoral in mg/dl")
fbs = st.number_input("Fasting Blood Sugar (1 for >120 mg/dl, 0 for <120 mg/dl)")
restecg = st.number_input("Resting Electrocardiographic Results (0 to 2)")
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.number_input("Exercise Induced Angina (1 for yes, 0 for no)")
oldpeak = st.number_input("ST Depression Induced by Exercise")
slope = st.number_input("Slope of the Peak Exercise ST Segment (0 to 2)")
ca = st.number_input("Number of Major Vessels (0 to 3)")
thal = st.number_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversable defect)")

# Show the prediction result
if st.button("Predict"):
    result = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if result >= 0.5:
        st.success("There is a high probability that the patient has heart disease")
    else:
        st.success("There is a low probability that the patient has heart disease")


# In[ ]:




