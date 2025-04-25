from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='.')
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins for testing

try:
    model = joblib.load('rainfall_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/')
def serve_index():
    print("Serving index.html")  # Log when index is requested
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Log input data
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                          'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing fields: {[f for f in required_fields if f not in data]}'}), 400

        try:
            input_data = {field: float(data[field]) for field in required_fields}
            print("Parsed input:", input_data)  # Log parsed input
        except (ValueError, TypeError) as e:
            print("Input parsing error:", str(e))  # Log parsing error
            return jsonify({'error': 'All fields must be numeric'}), 400

        input_df = pd.DataFrame([input_data], columns=required_fields)
        print("Input DataFrame:", input_df.to_dict())  # Log DataFrame
        input_scaled = scaler.transform(input_df)
        print("Scaled input:", input_scaled)  # Log scaled input
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        print("Prediction:", prediction, "Probability:", probability)  # Log output
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(float(probability), 4)
        })

    except Exception as e:
        print("Prediction error:", str(e))  # Log any other errors
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
