import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data):
    """
    Preprocess the kidney disease dataset
    
    Parameters:
    -----------
    data : pandas DataFrame
        The raw kidney disease dataset
    
    Returns:
    --------
    pandas DataFrame
        Preprocessed dataset ready for analysis and modeling
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Handling missing values
    # For categorical variables, fill with the most frequent value
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # For numerical variables, fill with the median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical to numerical
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'classification']
    
    for col in categorical_cols:
        # Map commonly used values
        if col == 'red_blood_cells' or col == 'pus_cell':
            df[col] = df[col].map({'normal': 0, 'abnormal': 1})
        elif col == 'pus_cell_clumbs' or col == 'bacteria':
            df[col] = df[col].map({'notpresent': 0, 'present': 1})
        elif col in ['hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'peda_edema', 'anemia']:
            df[col] = df[col].map({'no': 0, 'yes': 1})
        elif col == 'appetite':
            df[col] = df[col].map({'good': 0, 'poor': 1})
        else:
            # For any other categorical columns, use label encoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Process packed_cell_volume, white_blood_cell_count, red_blood_cell_count
    # Convert these to numeric if they're not already
    for col in ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # Process the target variable (classification)
    if 'classification' in df.columns:
        # Map 'ckd' to 1 (has disease) and 'notckd' to 0 (doesn't have disease)
        df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
        # If there are other values, convert them appropriately
        if df['classification'].isna().any():
            df['classification'] = df['classification'].fillna(1)  # Default to CKD if unknown
    
    return df

def prepare_user_input(user_input):
    """
    Prepares user input for prediction
    
    Parameters:
    -----------
    user_input : dict
        Dictionary containing user input for prediction
    
    Returns:
    --------
    pandas DataFrame
        Preprocessed user input ready for prediction
    """
    # Convert to DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Convert categorical to numerical
    categorical_mapping = {
        'red_blood_cells': {'normal': 0, 'abnormal': 1},
        'pus_cell': {'normal': 0, 'abnormal': 1},
        'pus_cell_clumbs': {'notpresent': 0, 'present': 1},
        'bacteria': {'notpresent': 0, 'present': 1},
        'hypertension': {'no': 0, 'yes': 1},
        'diabetes_mellitus': {'no': 0, 'yes': 1},
        'coronary_artery_disease': {'no': 0, 'yes': 1},
        'appetite': {'good': 0, 'poor': 1},
        'peda_edema': {'no': 0, 'yes': 1},
        'anemia': {'no': 0, 'yes': 1}
    }
    
    for col, mapping in categorical_mapping.items():
        if col in user_df.columns:
            user_df[col] = user_df[col].map(mapping)
    
    # Ensure all columns are numeric
    for col in user_df.columns:
        user_df[col] = pd.to_numeric(user_df[col], errors='coerce')
    
    return user_df
