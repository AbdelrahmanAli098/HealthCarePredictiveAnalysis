import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Processed data with numerical features
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the correlation heatmap
    """
    try:
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_num = df[numerical_cols].copy()
        
        # Handle missing values for correlation calculation
        df_num = df_num.fillna(df_num.mean())
        
        # Calculate correlation matrix
        corr_matrix = df_num.corr()
        
        # Create figure
        fig = Figure(figsize=(12, 10))
        ax = fig.subplots()
        
        # Create heatmap with improved aesthetics
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # To hide upper triangle
        cmap = sns.diverging_palette(220, 10, as_cmap=True)    # Blue to red color palette
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap=cmap, 
            vmax=1.0, 
            vmin=-1.0,
            center=0,
            linewidths=0.5, 
            ax=ax, 
            fmt=".2f", 
            annot_kws={"size": 8},
            cbar_kws={"shrink": .8}
        )
        
        ax.set_title('Correlation Heatmap of Numerical Features', fontsize=14)
        
        # Make the figure layout more appealing
        fig.tight_layout()
        
        return fig
    except Exception as e:
        # Create a figure with error message
        fig = Figure(figsize=(12, 10))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"Could not generate correlation heatmap: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def plot_feature_importance(model, df):
    """
    Plot feature importance from the model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    df : pandas DataFrame
        Processed data used for training
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the feature importance plot
    """
    try:
        # Get feature names (excluding target variable)
        features = df.drop('classification', axis=1).columns
        
        # Create figure
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models that have feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importance with more distinctive colors
            # Use a more vibrant and distinct color palette
            feature_colors = []
            for i, imp in enumerate(importances[indices]):
                # Make more important features stand out with deeper colors
                if imp > 0.1:  # High importance
                    feature_colors.append('#e74c3c')  # Bright red
                elif imp > 0.05:  # Medium importance 
                    feature_colors.append('#f39c12')  # Bright orange
                else:  # Low importance
                    feature_colors.append('#3498db')  # Bright blue
                    
            # Create bar chart with custom colors and thicker bars for better visibility
            bars = ax.bar(range(len(importances)), importances[indices], 
                         align='center', color=feature_colors, width=0.7)
            
            # Add value annotations to bars with improved visibility
            for i, bar in enumerate(bars):
                height = bar.get_height()
                # Add value labels with better contrast and visibility
                if importances[indices][i] >= 0.05:  # Add detailed labels for significant features
                    # Add white outline to text for better readability
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{importances[indices][i]:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold', color='black',
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none'))
            
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([features[i] for i in indices], rotation=90)
            # Improved chart title and styling
            ax.set_title('Random Forest Feature Importance', fontweight='bold', fontsize=14)
            ax.set_ylabel('Importance Score', fontweight='bold', fontsize=12)
            
            # Add a legend explaining the color coding
            legend_elements = [
                mpatches.Patch(color='#e74c3c', label='High Importance (>0.1)'),
                mpatches.Patch(color='#f39c12', label='Medium Importance (>0.05)'),
                mpatches.Patch(color='#3498db', label='Lower Importance (<0.05)')
            ]
            ax.legend(handles=legend_elements, loc='best', title='Feature Significance')
            
            # Enhanced grid and background style
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)  # Put grid lines behind bars
            ax.set_facecolor('#f8f9fa')
            
        else:
            # Try to use a safe feature importance approach for other models
            try:
                # Calculate feature importance
                X = df.drop('classification', axis=1)
                y = df['classification']
                
                # Try different approaches to get feature importance
                try:
                    # Use permutation importance with dictionary access
                    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
                    # Check if result is a dictionary or an object
                    if isinstance(result, dict):
                        if 'importances_mean' in result:
                            importances = result['importances_mean']
                        else:
                            # Fallback to a simple importance measure
                            importances = np.ones(X.shape[1]) / X.shape[1]
                    else:
                        importances = result.importances_mean
                except:
                    # Final fallback - calculate importance using correlations
                    importances = np.zeros(X.shape[1])
                    for i, col in enumerate(X.columns):
                        corr = abs(X[col].corr(pd.Series(y)))
                        importances[i] = 0.1 if np.isnan(corr) else corr
                
                # Sort importance - convert to numpy array first to avoid any issues
                importances_array = np.array(importances, dtype=float)
                sorted_idx = np.abs(importances_array).argsort()[::-1]
                
                # Plot with more distinctive colors for clarity
                feature_colors = []
                for i, imp in enumerate(importances_array[sorted_idx]):
                    # Make more important features stand out with deeper colors
                    if imp > 0.2:  # High importance
                        feature_colors.append('#e74c3c')  # Bright red
                    elif imp > 0.1:  # Medium importance 
                        feature_colors.append('#f39c12')  # Bright orange
                    elif imp > 0.05:  # Low-medium importance
                        feature_colors.append('#27ae60')  # Green
                    else:  # Low importance
                        feature_colors.append('#3498db')  # Bright blue
                
                # Create bar chart with custom colors and better spacing
                bars = ax.bar(range(X.shape[1]), importances_array[sorted_idx], 
                             color=feature_colors, width=0.7, alpha=0.9)
                
                ax.set_xticks(range(X.shape[1]))
                ax.set_xticklabels([features[i] for i in sorted_idx], rotation=90)
                # Improved chart title and styling
                ax.set_title('Feature Importance Analysis', fontweight='bold', fontsize=14)
                ax.set_ylabel('Importance Score', fontweight='bold', fontsize=12)
                
                # Add a legend explaining the color coding
                legend_elements = [
                    mpatches.Patch(color='#e74c3c', label='High Importance (>0.2)'),
                    mpatches.Patch(color='#f39c12', label='Medium Importance (>0.1)'),
                    mpatches.Patch(color='#27ae60', label='Low-Medium Importance (>0.05)'),
                    mpatches.Patch(color='#3498db', label='Low Importance (<0.05)')
                ]
                ax.legend(handles=legend_elements, loc='best', title='Feature Significance')
                
                # Enhanced grid and background style
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.set_axisbelow(True)  # Put grid lines behind bars
                ax.set_facecolor('#f8f9fa')
                
            except Exception as inner_e:
                ax.text(0.5, 0.5, f'Feature importance not available: {str(inner_e)}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        fig.tight_layout()
        return fig
        
    except Exception as e:
        # Create a figure with error message
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"Could not generate feature importance: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def plot_distribution_by_class(df, feature):
    """
    Plot distribution of a feature by class
    
    Parameters:
    -----------
    df : pandas DataFrame
        Processed data with numerical features
    feature : str
        Feature name to plot
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the distribution plot
    """
    # Create figure
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    
    # Plot distribution by class
    for class_value in df['classification'].unique():
        sns.kdeplot(
            df[df['classification'] == class_value][feature], 
            label=f"{'CKD' if class_value == 1 else 'Not CKD'}", 
            ax=ax,
            fill=True,
            alpha=0.3
        )
    
    ax.set_title(f'Distribution of {feature} by Kidney Disease Status')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_medical_parameters_comparison(df):
    """
    Plot comparison of key medical parameters between CKD and non-CKD patients
    
    Parameters:
    -----------
    df : pandas DataFrame
        Processed data with medical parameters
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the comparison plots
    """
    try:
        # Select key medical parameters
        medical_params = ['blood_pressure', 'blood_glucose_random', 'blood_urea', 
                          'serum_creatinine', 'hemoglobin', 'packed_cell_volume']
        
        # Create figure with subplots
        fig = Figure(figsize=(15, 10))
        axs = fig.subplots(2, 3)
        axs = axs.flatten()
        
        # Set a nicer color palette
        colors = ['#ff6b6b', '#48dbfb']  # Coral red and light blue
        
        # Plot each parameter
        for i, param in enumerate(medical_params):
            ax = axs[i]
            
            # Calculate means for each class with error handling
            ckd_data = df[df['classification'] == 1][param].dropna()
            not_ckd_data = df[df['classification'] == 0][param].dropna()
            
            ckd_mean = ckd_data.mean()
            not_ckd_mean = not_ckd_data.mean()
            
            # Calculate standard error for error bars
            ckd_std = ckd_data.std() / np.sqrt(len(ckd_data)) if len(ckd_data) > 0 else 0
            not_ckd_std = not_ckd_data.std() / np.sqrt(len(not_ckd_data)) if len(not_ckd_data) > 0 else 0
            
            # Create bar plot with error bars
            x_pos = np.arange(2)
            means = [ckd_mean, not_ckd_mean]
            errors = [ckd_std, not_ckd_std]
            
            bars = ax.bar(x_pos, means, yerr=errors, align='center', 
                         color=colors, alpha=0.7, ecolor='black', capsize=10)
            
            # Set axis labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['CKD', 'Not CKD'])
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Set title and labels with better formatting
            ax.set_title(f'Average {param.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Value', fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add background color to enhance readability
            ax.set_facecolor('#f8f9fa')
        
        fig.suptitle('Comparison of Medical Parameters Between CKD and Non-CKD Patients', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    except Exception as e:
        # Create a figure with error message
        fig = Figure(figsize=(15, 10))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"Could not generate medical parameters comparison: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def plot_pca_visualization(df):
    """
    Plot PCA visualization of the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Processed data with numerical features
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the PCA visualization
    """
    try:
        # Select numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'classification']
        
        # Create PCA model
        pca = PCA(n_components=2)
        
        # Handle missing values
        X = df[numerical_cols].copy()
        # Fill missing values with column means
        for col in X.columns:
            X[col] = X[col].fillna(X[col].mean())
        
        y = df['classification'].copy()
        
        # Fit and transform the data
        X_pca = pca.fit_transform(X)
        
        # Create figure
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()
        
        # Define colors for each class
        colors = ['#66b3ff', '#ff9999']
        cmap = ListedColormap(colors)
        
        # Plot PCA results
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, alpha=0.8, s=50)
        
        # Add legend
        legend_labels = ['Not CKD', 'CKD']
        legend = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Classification")
        
        # Add labels
        ax.set_title('PCA Visualization of Kidney Disease Data', fontsize=14)
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    except Exception as e:
        # Create a figure with error message
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"Could not generate PCA visualization: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def plot_feature_boxplots(df):
    """
    Plot boxplots for key features by classification
    
    Parameters:
    -----------
    df : pandas DataFrame
        Processed data with numerical features
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the boxplots
    """
    try:
        # Select key features for boxplots
        key_features = ['blood_glucose_random', 'blood_urea', 'serum_creatinine', 'hemoglobin']
        
        # Create figure with subplots
        fig = Figure(figsize=(15, 10))
        axs = fig.subplots(2, 2)
        axs = axs.flatten()
        
        # Create boxplots for each feature
        for i, feature in enumerate(key_features):
            if i < len(axs):
                ax = axs[i]
                
                # Create dataframes for each class
                ckd_data = df[df['classification'] == 1][feature].dropna()
                not_ckd_data = df[df['classification'] == 0][feature].dropna()
                
                # Create boxplot
                box_data = [not_ckd_data, ckd_data]
                ax.boxplot(box_data, labels=['Not CKD', 'CKD'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue'),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
                
                # Add labels
                ax.set_title(f'{feature.replace("_", " ").title()} by Classification')
                ax.set_ylabel('Value')
                ax.grid(True, linestyle='--', alpha=0.3)
        
        fig.suptitle('Boxplots of Key Features by Kidney Disease Classification', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    except Exception as e:
        # Create a figure with error message
        fig = Figure(figsize=(15, 10))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"Could not generate boxplots: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

def plot_age_gender_analysis(df):
    """
    Plot age-related analysis with demographics
    
    Parameters:
    -----------
    df : pandas DataFrame
        Processed data
    
    Returns:
    --------
    matplotlib Figure
        Figure containing the age analysis plots
    """
    try:
        # Create figure with subplots
        fig = Figure(figsize=(15, 6))
        axs = fig.subplots(1, 2)
        
        # Age distribution by class - with error handling
        ax1 = axs[0]
        
        # Drop NaN values
        age_data_not_ckd = df[df['classification'] == 0]['age'].dropna()
        age_data_ckd = df[df['classification'] == 1]['age'].dropna()
        
        # Create histograms for each class
        ax1.hist(age_data_not_ckd, bins=10, alpha=0.5, label='Not CKD', color='#66b3ff')
        ax1.hist(age_data_ckd, bins=10, alpha=0.5, label='CKD', color='#ff9999')
        
        # Add labels and legend
        ax1.set_title('Age Distribution by Kidney Disease Classification')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Age vs. Key Medical Parameter (e.g., Serum Creatinine)
        ax2 = axs[1]
        
        # Handle missing data
        subset_df = df[['age', 'serum_creatinine', 'classification']].dropna()
        
        # Create scatter plot
        scatter = ax2.scatter(subset_df['age'], subset_df['serum_creatinine'], 
                             c=subset_df['classification'], cmap=ListedColormap(['#66b3ff', '#ff9999']),
                             alpha=0.7, s=50)
        
        # Add labels and legend
        ax2.set_title('Age vs. Serum Creatinine by Classification')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Serum Creatinine')
        legend_labels = ['Not CKD', 'CKD']
        ax2.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        # Create a figure with error message
        fig = Figure(figsize=(15, 6))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"Could not generate age analysis: {str(e)}", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
