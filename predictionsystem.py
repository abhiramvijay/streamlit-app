import streamlit as st
import pandas as pd
import pickle
from collections import Counter
from sklearn import preprocessing
import pymongo
from pymongo import MongoClient


client = MongoClient()
client = MongoClient("mongodb+srv://abhiramvijaykumar:edcdpredictor@cluster0.w3jtrfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db=client.edcd
collection=db.patients

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
    st.title("Early Detection of Cardiovascular Decline")
    patient_number = st.text_input("Enter Patient Number")
    if st.button("Predict"):
        patient_data=collection.find_one({'p_id':patient_number})
        if patient_data is not None:
            url=patient_data['file_url']
            url='https://drive.google.com/uc?id='+url.split('/')[-2]
            data=pd.read_csv(url)

    # Upload CSV file
    # uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    # if uploaded_file is not None:
    #     data = pd.read_csv(uploaded_file)

    #     # Display uploaded data
    #     st.write("Uploaded Data:")
    #     st.write(data)

        # Load pre-trained model
            model_path = "fin_model2.sav"  # Replace with the path to your pre-trained model
            model = load_model(model_path)
    # url='https://drive.google.com/file/d/1VXHoENEoaiqsOAFKBQaWlbFh1Z664S_F/view?usp=sharing'
    # url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    # data = pd.read_csv(url)

    # Make predictions
            predictions = predict(model, data)
            prediction_counts=Counter(predictions)

    # Display predictions
            st.write("Predictions:")
            st.write("Patient No:- ",patient_data['p_id'])
            st.write("Patient Name:- ",patient_data['p_name'])
            st.write("Number of normal sinus rhythm: ",prediction_counts.get(1,0))
            st.write("Number of ventricular beats: ",prediction_counts.get(4,0))
        else:
            st.write("Patient not found.")

if __name__ == "__main__":
    main()
