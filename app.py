import streamlit as st
import numpy as np
import pickle
import sklearn 
# Print scikit-learn version for debugging
st.write(f"Scikit-learn version: {sklearn.__version__}")

# Load models
with open("dtr.pkl", "rb") as model_file:
    dtr = pickle.load(model_file)

with open("preprocessor.pkl", "rb") as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Streamlit App
st.title("Crop Yield Prediction")

# User input fields
Year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm/year)")
pesticides_tonnes = st.number_input("Pesticides Used (tonnes)")
avg_temp = st.number_input("Average Temperature (°C)")
Area = st.text_input("Area Name")
Item = st.text_input("Crop Name")

# Prediction button
if st.button("Predict Crop Yield"):
    try:
        # Prepare input data
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)

        # Make prediction
        prediction = dtr.predict(transformed_features)[0]

        # Display result
        st.success(f"Predicted Crop Yield: {prediction:.2f} tons")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
