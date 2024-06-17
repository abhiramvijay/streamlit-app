from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import logging
from bson import ObjectId

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Connect to MongoDB
try:
    client = MongoClient("mongodb+srv://abhiramvijaykumar:edcdpredictor@cluster0.w3jtrfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client.edcd
    collection = db.patients
    logging.info("Connected to MongoDB")
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {e}")

# Load the model
model_path = "fin_model2.sav"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Utility function to convert MongoDB documents to JSON serializable format
def convert_to_json_serializable(document):
    if isinstance(document, dict):
        for key, value in document.items():
            if isinstance(value, ObjectId):
                document[key] = str(value)
            elif isinstance(value, dict):
                convert_to_json_serializable(value)
            elif isinstance(value, list):
                document[key] = [convert_to_json_serializable(item) if isinstance(item, (dict, list)) else item for item in value]
    return document

# Utility function to convert DataFrame to JSON serializable format
def dataframe_to_dict(df):
    return {str(key): value for key, value in df.to_dict().items()}

# Endpoint to fetch patient data
@app.route('/get_patient_data', methods=['GET'])
def get_patient_data():
    try:
        patient_number = request.args.get('patient_number')
        logging.info(f"Fetching data for patient number: {patient_number}")
        patient_data = collection.find_one({'p_id': patient_number})
        
        if patient_data is not None:
            url = patient_data['file_url']
            url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
            
            response = requests.get(url)
            if response.status_code == 200:
                data = pd.read_csv(StringIO(response.text))
                logging.info(f"Patient data retrieved successfully for patient number: {patient_number}")
                patient_data = convert_to_json_serializable(patient_data)
                return jsonify({'status': 'success', 'data': dataframe_to_dict(data), 'patient_info': patient_data})
            else:
                logging.error(f"Failed to retrieve patient data file from URL: {url}")
                return jsonify({'status': 'error', 'message': 'Failed to retrieve patient data file.'}), 500
        else:
            logging.error(f"Patient not found for patient number: {patient_number}")
            return jsonify({'status': 'error', 'message': 'Patient not found.'}), 404
    except Exception as e:
        logging.error(f"Error in get_patient_data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = pd.DataFrame(request.json['data'])
        scaler = MinMaxScaler()
        modelsc = scaler.fit(data)
        scaled_data=modelsc.transform(data)
        predictions = model.predict(scaled_data)
        prediction_counts = Counter(predictions)
        
        logging.info("Prediction successful")
        return jsonify({
            'status': 'success',
            'predictions': {str(key): value for key, value in prediction_counts.items()}
        })
    except Exception as e:
        logging.error(f"Error in predict: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)