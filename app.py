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
model = joblib.load('rainfall_model.joblib')
scaler = joblib.load('scaler.joblib')

# Route to serve index.html for the root URL
@app.route('/')
def serve_index():
    return render_template('index.html')

# Route for prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                          'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Create input dataframe
        input_data = pd.DataFrame([data])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
