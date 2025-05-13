import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_and_preprocess_data(filepath='kidney_disease.csv'):
    df = pd.read_csv(filepath, index_col="id")
    
    # Rename columns for better readability
    df.columns = [
        "age", "blood_pressure", "specific_gravity", "albumin", "sugar", 
        "red_blood_cells", "pus_cell", "pus_cell_clumbs", "bacteria", "blood_glucose_random",
        "blood_urea", "serum_creatinine", "sodium", "potassium", "hemoglobin", "packed_cell_volume", 
        "white_blood_cell_count", "red_blood_cell_count", "hypertension", "diabetes_mellitus", 
        "coronary_artery_disease", "appetite", "peda_edema", "anemia", "classification"
    ]
    
    # Fill missing values
    df["packed_cell_volume"] = pd.to_numeric(df["packed_cell_volume"], errors="coerce")
    df["white_blood_cell_count"] = pd.to_numeric(df["white_blood_cell_count"], errors="coerce")
    df["red_blood_cell_count"] = pd.to_numeric(df["red_blood_cell_count"], errors="coerce")
    
    # Replace incorrect values
    df['diabetes_mellitus'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    df['coronary_artery_disease'].replace('\tno', 'no', inplace=True)
    df['classification'].replace({'ckd\t': 'ckd', 'notckd': 'not ckd'}, inplace=True)
    
    # Map classification to numeric
    df['classification'] = df['classification'].map({'ckd': 0, 'not ckd': 1})
    df['classification'] = pd.to_numeric(df['classification'], errors='coerce')
    
    # Fill nulls
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Plot numeric variable distributions
def plot_numeric_distributions(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 16))
    plotnumber = 1

    for col in num_cols:
        if plotnumber <= len(num_cols):
            plt.subplot(3, 5, plotnumber)
            sns.histplot(x=df[col], kde=True, color='royalblue')
            plt.xlabel(col)
        plotnumber += 1

    plt.suptitle('Distribution of Numeric Variables', fontsize=20, y=1)
    plt.tight_layout()
    return fig

# Plot categorical variable distributions
def plot_categorical_distributions(df):
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    fig = make_subplots(rows=(len(cat_cols) // 4) + 1, cols=4, subplot_titles=cat_cols)
    row = 1
    col = 1

    for cat_col in cat_cols:
        counts = df[cat_col].value_counts()
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts.values,
                text=counts.values,
                textposition='auto',
                marker=dict(color='steelblue'),
                name=cat_col
            ),
            row=row,
            col=col
        )
        col += 1
        if col > 4:
            col = 1
            row += 1

    fig.update_layout(
        height=800,
        width=1000,
        title_text="Distribution of Categorical Variables",
        showlegend=False
    )
    return fig