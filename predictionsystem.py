import streamlit as st
import pandas as pd
import pickle
from collections import Counter
from sklearn import preprocessing
import pymongo
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np



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
     # Load pre-trained model
            model_path = "fin_model2.sav"  
            model = load_model(model_path)
    # Make predictions
            predictions = predict(model, data)
            prediction_counts=Counter(predictions)

    # Display predictions
            st.write("Predictions:")
            st.write("Patient No:- ",patient_data['p_id'])
            st.write("Patient Name:- ",patient_data['p_name'])
            st.write("Number of normal sinus rhythm:- ",prediction_counts.get(1,0))
            st.write("Number of ventricular beats:- ",prediction_counts.get(4,0))
            st.write("Number of Superventricular beats:- ",prediction_counts.get(3,0))
            st.write("Number of fusion beats:- ",prediction_counts.get(0,0))
            st.write("Number of unclassifiable beats :- ",prediction_counts.get(2,0))

            # Debug: Print DataFrame information
            st.write("### Data Structure")
            st.write("DataFrame Columns:", data.columns.tolist())
            st.write("First few rows of data:")
            st.write(data.head())

            # Visualize the ECG data
            st.write("### ECG Recording")
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot using index as time and all columns as signals
            ax.plot(data.values)  # Plot all columns
            ax.set_title('ECG Signal')
            ax.set_xlabel('Sample Points')
            ax.set_ylabel('Amplitude')
            
            # Display the plot in Streamlit
            st.pyplot(fig)

            # Create visualization of different beat types
            st.write("### ECG Beat Types Visualization")
            
            # Dictionary to map prediction numbers to beat types
            beat_types = {
                1: "Normal Sinus Rhythm",
                4: "Ventricular Beat",
                3: "Superventricular Beat",
                0: "Fusion Beat",
                2: "Unclassifiable Beat"
            }
            
            # Get one example of each beat type that exists in predictions
            unique_beats = {}
            for i, pred in enumerate(predictions):
                if pred not in unique_beats and pred in beat_types:
                    unique_beats[pred] = data.iloc[i]

            if unique_beats:
                # Create subplots for each beat type found
                num_beats = len(unique_beats)
                fig, axs = plt.subplots(num_beats, 1, figsize=(12, 3*num_beats))
                if num_beats == 1:
                    axs = [axs]  # Make it iterable if only one subplot

                # Plot each beat type
                for i, (beat_type, beat_data) in enumerate(unique_beats.items()):
                    axs[i].plot(beat_data.values)
                    axs[i].set_title(f'{beat_types[beat_type]}')
                    axs[i].set_xlabel('Sample Points')
                    axs[i].set_ylabel('Amplitude')
                    
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No beat data available for visualization")
    
        else:
            st.write("Patient not found.")

if __name__ == "__main__":
    main()
