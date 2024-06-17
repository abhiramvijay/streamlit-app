import streamlit as st
import requests
import pandas as pd

# Main function to run the Streamlit app
def main():
    st.title("Early Detection of Cardiovascular Decline")
    
    patient_number = st.text_input("Enter Patient Number")
    
    if st.button("Predict"):
        try:
            # Fetch patient data
            response = requests.get("http://127.0.0.1:5000/get_patient_data", params={'patient_number': patient_number})
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            
            if data['status'] == 'success':
                patient_data = data['patient_info']
                patient_df = pd.DataFrame(data['data'])
                
                # Make prediction
                prediction_response = requests.post("http://127.0.0.1:5000/predict", json={'data': patient_df.to_dict()})
                prediction_response.raise_for_status()  # Raise an HTTPError for bad responses
                predictions = prediction_response.json()
                
                if predictions['status'] == 'success':
                    prediction_counts = predictions['predictions']
                    
                    # Display predictions
                    st.write("Predictions:")
                    st.write("Patient No:- ", patient_data['p_id'])
                    st.write("Patient Name:- ", patient_data['p_name'])
                    st.write("Number of normal sinus rhythm: ", prediction_counts.get('1', 0))
                    st.write("Number of ventricular beats: ", prediction_counts.get('4', 0))
                    st.write("Number of supraventricular beats: ", prediction_counts.get('3', 0))
                    st.write("Number of fusion beats: ", prediction_counts.get('0', 0))
                    st.write("Number of unclassifiable beats: ", prediction_counts.get('2', 0))
                else:
                    st.write("Error in prediction: ", predictions['message'])
            else:
                st.write("Error fetching patient data: ", data['message'])
        except requests.exceptions.RequestException as e:
            st.write("HTTP request failed: ", e)
        except ValueError as e:
            st.write("Error parsing JSON: ", e)
        except Exception as e:
            st.write("An unexpected error occurred: ", e)

if __name__ == "__main__":
    main()