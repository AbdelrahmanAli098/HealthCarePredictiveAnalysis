import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from utils.preprocessing import preprocess_data, prepare_user_input
from utils.visualization import (
    plot_correlation_heatmap, 
    plot_feature_importance, 
    plot_distribution_by_class,
    plot_medical_parameters_comparison,
    plot_pca_visualization,
    plot_feature_boxplots,
    plot_age_gender_analysis
)
from utils.model import train_model, load_model, predict
from notebook_utils import load_and_preprocess_data

# Ensure all features are numeric for PCA and feature importance
def preprocess_for_pca_and_importance(data):
    # Convert categorical columns to numeric using label encoding
    from sklearn.preprocessing import LabelEncoder
    data = data.copy()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    return data

# Set page configuration
st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def get_data():
    raw_data = pd.read_csv('kidney_disease.csv', index_col="id")
    processed_data = load_and_preprocess_data('kidney_disease.csv')
    return raw_data, processed_data

raw_data, processed_data = get_data()

# App title and description
st.title("Kidney Disease Prediction App")
st.markdown("""
This application helps predict the likelihood of Chronic Kidney Disease (CKD) based on patient's medical parameters.
It also provides insights and visualizations to better understand the dataset.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Prediction", "ℹ️ About Kidney Disease"])

# Tab 1: Data Overview
with tab1:
    st.header("Dataset Overview")
    
    # Raw dataset
    st.subheader("Raw Dataset (With Missing Values)")
    st.dataframe(raw_data.head())
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_data = raw_data.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Values']
    missing_data['Missing Percentage'] = (missing_data['Missing Values'] / len(raw_data)) * 100
    missing_data = missing_data.sort_values('Missing Percentage', ascending=False)
    st.dataframe(missing_data)
    
    # Preprocessed dataset
    st.subheader("Preprocessed Dataset (After Handling Missing Values)")
    st.dataframe(processed_data.head())
    
    # Class distribution
    st.subheader("Class Distribution")
    if 'classification' in processed_data.columns:
        class_dist = processed_data['classification'].value_counts().reset_index()
        class_dist.columns = ['Class', 'Count']
        fig = px.pie(class_dist, values='Count', names='Class', 
                     title='Distribution of Kidney Disease Classes',
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig)
    
    # Insights
    st.markdown("""
    ### Insights:
    - The raw dataset contains missing values, which have been handled during preprocessing.
    - The dataset is slightly imbalanced, with more instances of one class than the other.
    """)

# Tab 2: Visualizations
with tab2:
    st.header("Data Visualizations")
    
    if not processed_data.empty:
        try:
            # Create tabs for different visualization types
            viz_tabs = st.tabs(["📊 Correlations", "📈 Distributions", "👥 Demographics", "🔍 Advanced Analysis", "📊 Additional Plots"])
            
            # Correlation Heatmap and Medical Parameters Comparison
            with viz_tabs[0]:
                st.subheader("Correlation Between Features")
                fig_corr = plot_correlation_heatmap(processed_data)
                st.pyplot(fig_corr)
                
                st.markdown("""
                ### Insights:
                - Strong correlations exist between **packed cell volume**, **hemoglobin**, and CKD classification.
                - **Serum creatinine** and **blood urea** are also highly correlated with CKD.
                - Features with high correlations can provide valuable information for predicting CKD.
                """)

                st.subheader("Medical Parameters Comparison")
                fig_medical = plot_medical_parameters_comparison(processed_data)
                st.pyplot(fig_medical)
            
            # Feature Distributions and Boxplots
            with viz_tabs[1]:
                # Feature distributions by class
                st.subheader("Feature Distributions by Class")
                
                # Choose numerical columns for distribution plots
                numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'classification']
                
                selected_feature = st.selectbox(
                    "Select a feature to visualize its distribution",
                    options=numeric_cols
                )
                
                if selected_feature:
                    fig_dist = plot_distribution_by_class(processed_data, selected_feature)
                    st.pyplot(fig_dist)
                
                # Feature boxplots
                st.subheader("Key Feature Boxplots")
                fig_boxplots = plot_feature_boxplots(processed_data)
                st.pyplot(fig_boxplots)
                
                st.markdown("""
                ### Insights:
                - Features like **serum creatinine** and **blood urea** show clear separation between CKD and non-CKD classes.
                - This separation indicates that these features are strong predictors of CKD.
                - Other features, such as **hemoglobin**, also show distinct distributions for the two classes.
                """)
            
            with viz_tabs[2]:
                # Age-related analysis
                st.subheader("Age-Related Analysis")
                fig_age = plot_age_gender_analysis(processed_data)
                st.pyplot(fig_age)
                
                # Add description about demographics
                st.markdown("""
                ### Demographic Insights
                The charts above show how age relates to kidney disease classification and serum creatinine levels.
                Patterns in these visualizations can help identify at-risk age groups and guide targeted screening efforts.
                """)
                
                st.markdown("""
                ### Insights:
                - CKD is more prevalent in older age groups, as seen in the age distribution.
                - Younger individuals are less likely to have CKD, but exceptions exist.
                - Age-related patterns can help identify at-risk populations for targeted screening.
                """)
            
            with viz_tabs[3]:
                # PCA Visualization
                st.subheader("PCA Visualization")
                try:
                    numeric_data = preprocess_for_pca_and_importance(processed_data)
                    fig_pca = plot_pca_visualization(numeric_data)
                    st.pyplot(fig_pca)
                    
                    st.markdown("""
                    ### Principal Component Analysis (PCA)
                    PCA reduces the dimensionality of the data while preserving as much variance as possible.
                    This visualization shows how the data points are distributed in a 2D space created from all numerical features.
                    Points that cluster together have similar characteristics, and the separation between red and blue points
                    indicates how well the features can distinguish between CKD and non-CKD cases.
                    """)
                except Exception as e:
                    st.error(f"Error in PCA visualization: {str(e)}")
                
                # Feature Importance
                st.subheader("Feature Importance")
                try:
                    numeric_data = preprocess_for_pca_and_importance(processed_data)
                    model = train_model(numeric_data)
                    fig_importance = plot_feature_importance(model, numeric_data)
                    st.pyplot(fig_importance)
                    
                    st.markdown("""
                    ### Feature Importance Interpretation
                    This chart shows which medical parameters have the strongest influence on the model's predictions.
                    Parameters with higher importance values have a greater impact on determining whether a patient
                    has chronic kidney disease.
                    """)
                except Exception as imp_error:
                    st.error(f"Error generating feature importance: {str(imp_error)}")
            
            # Additional Plots from Notebook
            with viz_tabs[4]:
                st.subheader("Additional Plots")
                
                # Plot 1: Distribution of Numeric Variables
                st.subheader("Distribution of Numeric Variables")
                fig_numeric = plt.figure(figsize=(12, 16))
                plotnumber = 1
                for col in numeric_cols:
                    if plotnumber <= len(numeric_cols):
                        ax = plt.subplot(3, 5, plotnumber)
                        sns.histplot(x=processed_data[col], kde=True, color='royalblue')
                        plt.xlabel(col)
                    plotnumber += 1
                plt.tight_layout()
                st.pyplot(fig_numeric)
                
                st.markdown("""
                ### Insights:
                - Most numeric variables show a skewed distribution, indicating potential outliers.
                - Variables like **serum creatinine** and **blood urea** have high values for CKD patients.
                """)

                # Plot 2: Distribution of Categorical Variables
                st.subheader("Distribution of Categorical Variables")
                cat_cols = processed_data.select_dtypes(include=['object']).columns.tolist()
                fig_categorical = make_subplots(rows=(len(cat_cols) // 4) + 1, cols=4, subplot_titles=cat_cols)
                row = 1
                col = 1
                for cat_col in cat_cols:
                    counts = processed_data[cat_col].value_counts()
                    fig_categorical.add_trace(
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
                fig_categorical.update_layout(
                    height=800,
                    width=1000,
                    title_text="Distribution of Categorical Variables",
                    showlegend=False
                )
                st.plotly_chart(fig_categorical)
                
                st.markdown("""
                ### Insights:
                - Categorical variables like **pus cell** and **red blood cells** show clear differences between CKD and non-CKD patients.
                - The presence of bacteria and pus cell clumps is more common in CKD patients.
                - These variables provide valuable information for diagnosing CKD.
                """)
                
                # Add more plots from your notebook as needed
                st.subheader("Scatter Plot: Serum Creatinine vs. Blood Urea")
                fig_scatter = plt.figure()
                sns.scatterplot(
                    x=processed_data["serum_creatinine"], 
                    y=processed_data["blood_urea"], 
                    hue=processed_data["classification"], 
                    palette="coolwarm"
                )
                plt.title("Serum Creatinine vs. Blood Urea by CKD Status")
                st.pyplot(fig_scatter)
                
                st.markdown("""
                ### Insights:
                - Higher levels of **serum creatinine** and **blood urea** are strongly associated with CKD.
                - Non-CKD patients tend to have lower values for both features.
                - This scatter plot highlights the importance of these two features in distinguishing between CKD and non-CKD cases.
                """)

        except Exception as e:
            st.error(f"Error in data visualization: {str(e)}")
    else:
        st.warning("No data available for visualization.")

with tab3:
    st.header("Kidney Disease Prediction")
    
    if not processed_data.empty:
        st.markdown("""
        Enter the patient's medical parameters to predict the likelihood of Chronic Kidney Disease.
        """)
        
        # Split form into multiple columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, value=50)
            blood_pressure = st.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=200, value=80)
            specific_gravity = st.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=2)
            albumin = st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5], index=0)
            sugar = st.selectbox("Sugar", options=[0, 1, 2, 3, 4, 5], index=0)
            red_blood_cells = st.selectbox("Red Blood Cells", options=["normal", "abnormal"], index=0)
            pus_cell = st.selectbox("Pus Cell", options=["normal", "abnormal"], index=0)
            pus_cell_clumbs = st.selectbox("Pus Cell Clumps", options=["present", "notpresent"], index=1)
        
        with col2:
            bacteria = st.selectbox("Bacteria", options=["present", "notpresent"], index=1)
            blood_glucose_random = st.number_input("Blood Glucose Random (mgs/dl)", min_value=70, max_value=500, value=120)
            blood_urea = st.number_input("Blood Urea (mgs/dl)", min_value=10, max_value=200, value=40)
            serum_creatinine = st.number_input("Serum Creatinine (mgs/dl)", min_value=0.4, max_value=15.0, value=1.2, step=0.1)
            sodium = st.number_input("Sodium (mEq/L)", min_value=100, max_value=170, value=135)
            potassium = st.number_input("Potassium (mEq/L)", min_value=2.5, max_value=7.5, value=4.0, step=0.1)
            hemoglobin = st.number_input("Hemoglobin (gms)", min_value=3.0, max_value=18.0, value=12.0, step=0.1)
            
        with col3:
            packed_cell_volume = st.number_input("Packed Cell Volume", min_value=15, max_value=55, value=40)
            white_blood_cell_count = st.number_input("White Blood Cell Count (cells/cumm)", min_value=2000, max_value=25000, value=9000)
            red_blood_cell_count = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
            hypertension = st.selectbox("Hypertension", options=["yes", "no"], index=1)
            diabetes_mellitus = st.selectbox("Diabetes Mellitus", options=["yes", "no"], index=1)
            coronary_artery_disease = st.selectbox("Coronary Artery Disease", options=["yes", "no"], index=1)
            appetite = st.selectbox("Appetite", options=["good", "poor"], index=0)
            peda_edema = st.selectbox("Pedal Edema", options=["yes", "no"], index=1)
            anemia = st.selectbox("Anemia", options=["yes", "no"], index=1)
        
        # Create a dictionary with user inputs
        user_input = {
            'age': age,
            'blood_pressure': blood_pressure,
            'specific_gravity': specific_gravity,
            'albumin': albumin,
            'sugar': sugar,
            'red_blood_cells': red_blood_cells,
            'pus_cell': pus_cell,
            'pus_cell_clumbs': pus_cell_clumbs,
            'bacteria': bacteria,
            'blood_glucose_random': blood_glucose_random,
            'blood_urea': blood_urea,
            'serum_creatinine': serum_creatinine,
            'sodium': sodium,
            'potassium': potassium,
            'hemoglobin': hemoglobin,
            'packed_cell_volume': packed_cell_volume,
            'white_blood_cell_count': white_blood_cell_count,
            'red_blood_cell_count': red_blood_cell_count,
            'hypertension': hypertension,
            'diabetes_mellitus': diabetes_mellitus,
            'coronary_artery_disease': coronary_artery_disease,
            'appetite': appetite,
            'peda_edema': peda_edema,
            'anemia': anemia
        }
        
        # Prediction button
        if st.button("Predict"):
            try:
                # Preprocess user input
                user_input_df = pd.DataFrame([user_input])
                user_input_df = preprocess_data(user_input_df)  # Ensure this matches your preprocessing logic
                
                # Load or train the model
                try:
                    model = load_model()
                except:
                    model = train_model(processed_data)
                
                # Make prediction
                prediction = model.predict(user_input_df)
                prediction_proba = model.predict_proba(user_input_df)
                
                # Display prediction
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.success("The patient is likely NOT to have Chronic Kidney Disease (CKD).")
                else:
                    st.error("The patient is likely to have Chronic Kidney Disease (CKD).")
                
                # Display prediction probabilities
                st.subheader("Prediction Probabilities")
                st.write(f"Probability of NOT CKD: {prediction_proba[0][1]:.2f}")
                st.write(f"Probability of CKD: {prediction_proba[0][0]:.2f}")
            
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    else:
        st.warning("No data available for model training. Please ensure the dataset is correctly loaded.")

# Tab 4: About Kidney Disease
with tab4:
    st.header("About Chronic Kidney Disease")
    
    st.markdown("""
    ## What is Chronic Kidney Disease (CKD)?
    
    Chronic Kidney Disease is a condition characterized by a gradual loss of kidney function over time.
    The kidneys filter wastes and excess fluids from the blood, which are then excreted in the urine.
    When CKD reaches an advanced stage, dangerous levels of fluid, electrolytes, and wastes can build up in the body.
    
    ## Risk Factors for CKD
    
    - **Diabetes**: Leading cause of kidney disease
    - **High Blood Pressure**: Second leading cause of kidney disease
    - **Heart Disease**: Can reduce blood flow to the kidneys
    - **Family History**: Genetic factors can increase risk
    - **Age**: Risk increases with age
    - **Obesity**: Can contribute to diabetes and high blood pressure
    
    ## Key Medical Parameters in CKD Diagnosis
    
    ### Blood Tests
    - **Blood Urea Nitrogen (BUN)**: Measures the amount of urea nitrogen in blood
    - **Serum Creatinine**: Waste product filtered by kidneys
    - **Glomerular Filtration Rate (GFR)**: Calculated from creatinine level, age, body size, and gender
    
    ### Urine Tests
    - **Albumin**: Protein that can appear in urine when kidneys are damaged
    - **Specific Gravity**: Concentration of particles in urine
    - **Red Blood Cells and White Blood Cells**: Can indicate infection or inflammation
    
    ### Other Parameters
    - **Blood Pressure**: Hypertension can both cause and result from kidney disease
    - **Blood Glucose**: Diabetes is a major risk factor for kidney disease
    - **Hemoglobin**: Kidneys produce a hormone that triggers red blood cell production
    
    ## Normal Ranges for Key Features
    
    | Feature                     | Normal Range Value             |
    | --------------------------- | ------------------------------ |
    | **Age**                     | 18 – 60 (CKD risk increases >60) |
    | **Blood Pressure (mmHg)**   | 90 – 120 (systolic)            |
    | **Specific Gravity**        | 1.015 – 1.030                  |
    | **Albumin**                 | 0 (no protein in urine)        |
    | **Sugar**                   | 0 (no sugar in urine)          |
    | **Red Blood Cells**         | 1 (normal), 0 = abnormal       |
    | **Pus Cell**                | 1 (normal), 0 = abnormal       |
    | **Pus Cell Clumps**         | 0 (absent, normal), 1 = present |
    | **Bacteria**                | 0 (absent, normal), 1 = present |
    | **Blood Glucose Random**    | 70 – 140 mg/dL                 |
    | **Blood Urea**              | 7 – 20 mg/dL                   |
    | **Serum Creatinine**        | 0.6 – 1.2 mg/dL                |
    | **Sodium**                  | 135 – 145 mEq/L                |
    | **Potassium**               | 3.5 – 5.0 mEq/L                |
    | **Hemoglobin**              | 13.0 – 17.0 g/dL               |
    | **Packed Cell Volume**      | 36 – 50%                       |
    | **White Blood Cell Count**  | 5,000 – 11,000 cells/cumm      |
    | **Red Blood Cell Count**    | 4.5 – 6.0 million/cmm          |
    | **Hypertension**            | 0 = No, 1 = Yes                |
    | **Diabetes Mellitus**       | 0 = No, 1 = Yes                |
    | **Coronary Artery Disease** | 0 = No, 1 = Yes                |
    | **Appetite**                | 1 = Good (normal), 0 = Poor    |
    | **Peda Edema**              | 0 = Absent (normal), 1 = Present |
    | **Anemia**                  | 0 = No (normal), 1 = Yes       |
    
    ## Prevention and Management
    
    - **Control Blood Pressure**: Keep below 140/90 mm Hg
    - **Manage Diabetes**: Keep blood glucose within target range
    - **Healthy Diet**: Low in sodium, processed foods, and red meat
    - **Regular Exercise**: At least 30 minutes of moderate activity most days
    - **Avoid Smoking and Excessive Alcohol**: Both can damage kidneys
    - **Regular Check-ups**: Monitor kidney function if at risk
    
    ## Stages of CKD
    
    CKD is classified into five stages based on GFR:
    
    1. **Stage 1**: Kidney damage with normal GFR (≥90 mL/min)
    2. **Stage 2**: Mild reduction in GFR (60-89 mL/min)
    3. **Stage 3**: Moderate reduction in GFR (30-59 mL/min)
    4. **Stage 4**: Severe reduction in GFR (15-29 mL/min)
    5. **Stage 5**: Kidney failure (GFR <15 mL/min or on dialysis)
    
    Early detection and treatment can often prevent CKD from getting worse.
    """)

# Main script execution
if __name__ == "__main__":
    # This will run the entire app when executed
    pass

