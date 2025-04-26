from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/predict": {"origins": "*"}})

try:
    model_path = os.path.join(os.path.dirname(__file__), 'rainfall_model.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler file missing: {model_path}, {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    raise

@app.route('/')
def serve_index():
    print("Serving index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                          'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing fields: {[f for f in required_fields if f not in data]}'}), 400

        try:
            input_data = {field: float(data[field]) for field in required_fields}
            print("Parsed input:", input_data)
        except (ValueError, TypeError) as e:
            print("Input parsing error:", str(e))
            return jsonify({'error': 'All fields must be numeric'}), 400

        input_df = pd.DataFrame([input_data], columns=required_fields)
        print("Input DataFrame:", input_df.to_dict())
        input_scaled = scaler.transform(input_df)
        print("Scaled input:", input_scaled)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        print("Prediction:", prediction, "Probability:", probability)
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability': round(float(probability), 4)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)