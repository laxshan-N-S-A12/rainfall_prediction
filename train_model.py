import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
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
        raise ValueError("Invalid or missing values in 'rainfall' column after mapping")

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    X = df[features]
    y = df['rainfall']

    return X, y

def train_model():
    try:
        df = pd.read_csv('Rainfall.csv')
        print("Dataset loaded successfully. Shape:", df.shape)

        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Training set shape:", X_train.shape)
        print("Test set shape:", X_test.shape)

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

        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        print(f"Training accuracy: {train_accuracy:.2f}")
        print(f"Test accuracy: {test_accuracy:.2f}")
        print(f"Cross-validation scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, model.predict(X_test_scaled)))
        print("\nConfusion Matrix (Test Set):")
        print(confusion_matrix(y_test, model.predict(X_test_scaled)))

        joblib.dump(model, 'rainfall_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        print("Model and scaler saved successfully.")

        return model, scaler

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    model, scaler = train_model()
