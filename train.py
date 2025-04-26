import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()
    required_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                        'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in dataset")

    df['rainfall'] = df['rainfall'].str.lower().map({'yes': 1, 'no': 0})
    if df['rainfall'].isnull().any():
        raise ValueError("Invalid or missing values in 'rainfall' column")

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    X = df[features]
    y = df['rainfall']
    return X, y, features

def train_model():
    try:
        df = pd.read_csv('Rainfall.csv')
        print("Dataset loaded. Shape:", df.shape)

        X, y, features = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        print(f"Training accuracy: {model.score(X_train_scaled, y_train):.2f}")
        print(f"Test accuracy: {model.score(X_test_scaled, y_test):.2f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, model.predict(X_test_scaled)))

        joblib.dump(model, 'rainfall_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        print("Model and scaler saved.")

        # Test prediction
        test_input = {
            'pressure': 1013.2,
            'maxtemp': 25.0,
            'temparature': 20.0,
            'mintemp': 15.0,
            'dewpoint': 18.0,
            'humidity': 50,
            'cloud': 20,
            'sunshine': 7.5,
            'winddirection': 180,
            'windspeed': 10.5
        }
        test_df = pd.DataFrame([test_input], columns=features)
        test_scaled = scaler.transform(test_df)
        prediction = model.predict(test_scaled)[0]
        probability = model.predict_proba(test_scaled)[0][1]
        print(f"\nTest Prediction: {'Yes' if prediction == 1 else 'No'}, Probability: {probability:.4f}")

        return model, scaler

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()