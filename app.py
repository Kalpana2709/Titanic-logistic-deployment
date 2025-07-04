
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('logistic_model.pkl')

st.title("Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0)
fare = st.number_input("Fare", min_value=0.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# Create input DataFrame
input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.subheader(f"Prediction: {result}")
