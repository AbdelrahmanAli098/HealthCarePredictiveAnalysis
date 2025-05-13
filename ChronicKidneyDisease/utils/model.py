import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(data):
    """
    Train a machine learning model on the kidney disease dataset
    
    Parameters:
    -----------
    data : pandas DataFrame
        Preprocessed kidney disease data
    
    Returns:
    --------
    sklearn model
        Trained model ready for predictions
    """
    # Split data into features and target
    X = data.drop('classification', axis=1)
    y = data['classification']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model (good baseline model)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save the model
    save_model(model)
    
    return model

def save_model(model):
    """
    Save the trained model to disk
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to be saved
    """
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    with open('models/kidney_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model():
    """
    Load the trained model from disk
    
    Returns:
    --------
    sklearn model
        Loaded model ready for predictions
    """
    # Check if model file exists
    if not os.path.exists('models/kidney_disease_model.pkl'):
        raise FileNotFoundError("Model file not found. Please train the model first.")
    
    # Load the model
    with open('models/kidney_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict(model, user_input):
    """
    Make prediction using the trained model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model for prediction
    user_input : pandas DataFrame
        Preprocessed user input data
    
    Returns:
    --------
    tuple
        (predicted_class, probability)
    """
    # Make prediction
    pred_proba = model.predict_proba(user_input)[0]
    pred_class = model.predict(user_input)[0]
    
    # Get the probability for the predicted class
    if pred_class == 1:  # CKD
        probability = pred_proba[1]
    else:  # Not CKD
        probability = pred_proba[0]
    
    return pred_class, probability
