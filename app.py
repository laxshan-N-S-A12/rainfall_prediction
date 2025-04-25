from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# Initialize Flask app with template folder set to current directory
app = Flask(__name__, template_folder='.')
CORS(app)

# Load model and scaler
try:
    model = joblib.load('rainfall_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    raise

# Route to serve index.html for the root URL
@app.route('/')
def serve_index():
    return render_template('index.html')

# Route for prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate input
        required_fields = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                          'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing fields: {[f for f in required_fields if f not in data]}'}), 400

        # Convert input to float and create dataframe
        try:
            input_data = {field: float(data[field]) for field in required_fields}
        except (ValueError, TypeError):
            return jsonify({'error': 'All fields must be numeric'}), 400

        input_df = pd.DataFrame([input_data], columns=required_fields)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
