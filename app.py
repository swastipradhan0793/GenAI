import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st
import pickle

# import os
# print("Current directory:", os.getcwd())
# print("Files:", os.listdir())

# Load the model
model = load_model('Modeling/model.h5')

# Load the encoder and scaler 
with open('Modeling/lable_encoder_gender.pkl', 'rb') as file:
    lable_encoder_geo = pickle.load(file)

with open('Modeling/onehot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open("Modeling/scaler_X_train.pkl", 'rb') as file:
    scaler = pickle.load(file)


### Streamlit app
st.title("Customer Churn Prediction")

# User input

geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', lable_encoder_geo.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#Input data
input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [lable_encoder_geo.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})

# onehot encoded Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine 
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)

#Scale the data
input_data_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability:{prediction_prob:.2f}')
if prediction_prob > 0.5:
 st.write("Customer is likely to churn")
else:
 st.write("Customer is not likely to churn")