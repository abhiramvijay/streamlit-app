import streamlit as st
import pandas as pd
import pickle
from collections import Counter
from sklearn import preprocessing

# Function to load the pre-trained machine learning model
def load_model(model_path):
    # Load the model using pickle
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions using the loaded model
def predict(model, data):
    # Assuming your model accepts pandas DataFrame as input
    scaler=preprocessing.MinMaxScaler()
    modelsc=scaler.fit(data)
    scaled_data=modelsc.transform(data)
    predictions = model.predict(scaled_data)
    return predictions
    

# Main function to run the Streamlit app
def main():
    st.title("Upload CSV and Predict")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.write("Uploaded Data:")
        st.write(data)

        # Load pre-trained model
        model_path = "fin_model2.sav"  # Replace with the path to your pre-trained model
        model = load_model(model_path)

        # Make predictions
        predictions = predict(model, data)
        prediction_counts=Counter(predictions)

        # Display predictions
        st.write("Predictions:")
        st.write("Number of normal sinus rhythm: ",prediction_counts.get(1,0))
        st.write("Number of ventricular beats: ",prediction_counts.get(4,0))

if __name__ == "__main__":
    main()
