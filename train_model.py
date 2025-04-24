import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def preprocess_data(df):
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Convert rainfall to binary (1 for yes, 0 for no)
    df['rainfall'] = df['rainfall'].map({'yes': 1, 'no': 0})
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Select features using exact column names
    features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    
    # Verify all features exist in the dataframe
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in dataset")
    
    X = df[features]
    y = df['rainfall']
    
    return X, y

# Train model
def train_model():
    # Read data
    df = pd.read_csv('Rainfall.csv')
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'rainfall_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Evaluate model
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = train_model()
