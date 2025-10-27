#!/usr/bin/env python3
"""
BigMart Sales Prediction - Main Analysis Script

This script performs comprehensive analysis of BigMart sales data including:
- Data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Model training and evaluation
- Results visualization

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bigmart_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BigMartSalesAnalyzer:
    """
    Comprehensive BigMart sales analysis and prediction system.
    """
    
    def __init__(self, data_url: str = None):
        """
        Initialize the analyzer.
        
        Args:
            data_url: URL to the training data
        """
        self.data_url = data_url or 'https://raw.githubusercontent.com/GowsalyaKN/BigMart-Sales-Prediction/main/data/Train.csv'

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the BigMart sales dataset.
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info("Loading BigMart sales dataset...")
            self.df = pd.read_csv(self.data_url)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.
        
        Returns:
            Dictionary containing EDA results
        """
        logger.info("Performing exploratory data analysis...")
        
        eda_results = {
            'shape': self.df.shape,
            'info': self.df.info(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_summary': self.df.describe().to_dict(),
            'categorical_columns': [],
            'numeric_columns': []
        }
        
        # Identify column types
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                eda_results['categorical_columns'].append(col)
            else:
                eda_results['numeric_columns'].append(col)
        
        logger.info(f"EDA completed. Found {len(eda_results['categorical_columns'])} categorical and {len(eda_results['numeric_columns'])} numeric columns.")
        return eda_results
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values in the dataset.
        """
        logger.info("Handling missing values...")
        
        # Handle Item_Weight missing values
        if self.df['Item_Weight'].isnull().sum() > 0:
            item_weight_mean = self.df.pivot_table(values="Item_Weight", index='Item_Identifier')
            miss_bool = self.df['Item_Weight'].isnull()
            
            for i, item in enumerate(self.df['Item_Identifier']):
                if miss_bool[i]:
                    if item in item_weight_mean.index:
                        self.df.loc[i, 'Item_Weight'] = item_weight_mean.loc[item]['Item_Weight']
                    else:
                        self.df.loc[i, 'Item_Weight'] = self.df['Item_Weight'].mean()
        
        # Handle Outlet_Size missing values
        if self.df['Outlet_Size'].isnull().sum() > 0:
            outlet_size_mode = self.df.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Medium')
            miss_bool = self.df['Outlet_Size'].isnull()
            
            for outlet_type in self.df.loc[miss_bool, 'Outlet_Type'].unique():
                mode_size = outlet_size_mode.get(outlet_type, 'Medium')
                self.df.loc[(miss_bool) & (self.df['Outlet_Type'] == outlet_type), 'Outlet_Size'] = mode_size
        
        logger.info("Missing values handled successfully.")
    
    def feature_engineering(self) -> None:
        """
        Perform feature engineering on the dataset.
        """
        logger.info("Performing feature engineering...")
        
        # Handle Item_Visibility zeros
        zero_visibility = (self.df['Item_Visibility'] == 0).sum()
        if zero_visibility > 0:
            self.df.loc[self.df['Item_Visibility'] == 0, 'Item_Visibility'] = self.df['Item_Visibility'].mean()
            logger.info(f"Replaced {zero_visibility} zero visibility values with mean")
        
        # Standardize Item_Fat_Content
        fat_content_mapping = {
            'LF': 'Low Fat',
            'reg': 'Regular',
            'low fat': 'Low Fat'
        }
        self.df['Item_Fat_Content'] = self.df['Item_Fat_Content'].replace(fat_content_mapping)
        
        # Create New_Item_Type from Item_Identifier
        self.df['New_Item_Type'] = self.df['Item_Identifier'].apply(lambda x: x[:2])
        item_type_mapping = {
            'FD': 'Food',
            'NC': 'Non-Consumable',
            'DR': 'Drinks'
        }
        self.df['New_Item_Type'] = self.df['New_Item_Type'].map(item_type_mapping)
        
        # Handle fat content for non-consumable items
        self.df.loc[self.df['New_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
        
        # Create Outlet_Years feature
        self.df['Outlet_Years'] = 2013 - self.df['Outlet_Establishment_Year']
        
        logger.info("Feature engineering completed successfully.")
    
    def encode_categorical_features(self) -> None:
        """
        Encode categorical features for machine learning.
        """
        logger.info("Encoding categorical features...")
        
        # Label encode outlet identifier
        le = LabelEncoder()
        self.df['Outlet'] = le.fit_transform(self.df['Outlet_Identifier'])
        
        # Categorical columns for encoding
        cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                   'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
        
        # Label encode categorical columns
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col])
        
        # One-hot encode specific columns
        one_hot_cols = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 
                       'Outlet_Type', 'New_Item_Type']
        self.df = pd.get_dummies(self.df, columns=one_hot_cols, drop_first=True)
        
        logger.info("Categorical encoding completed.")
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Returns:
            Tuple of features and target
        """
        logger.info("Preparing features for modeling...")
        
        # Columns to drop
        drop_cols = ['Outlet_Establishment_Year', 'Item_Identifier', 
                    'Outlet_Identifier', 'Item_Outlet_Sales']
        
        X = self.df.drop(columns=[col for col in drop_cols if col in self.df.columns])
        y = self.df['Item_Outlet_Sales']
        
        # Log transform target variable
        y = np.log(1 + y)
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set
        """
        logger.info("Splitting data into train and test sets...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train multiple machine learning models.
        
        Returns:
            Dictionary containing model results
        """
        logger.info("Training machine learning models...")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Lasso Regression': Lasso(alpha=0.01, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models = models
        self.results = results
        return results
    
    def create_visualizations(self) -> None:
        """
        Create comprehensive visualizations for the analysis.
        """
        logger.info("Creating visualizations...")
        
        # Create results directory
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        # 1. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Distributions', fontsize=16)
        
        # Item Weight distribution
        sns.histplot(self.df['Item_Weight'].dropna(), ax=axes[0,0], kde=True)
        axes[0,0].set_title('Item Weight Distribution')
        
        # Item Visibility distribution
        sns.histplot(self.df['Item_Visibility'], ax=axes[0,1], kde=True)
        axes[0,1].set_title('Item Visibility Distribution')
        
        # Item MRP distribution
        sns.histplot(self.df['Item_MRP'], ax=axes[1,0], kde=True)
        axes[1,0].set_title('Item MRP Distribution')
        
        # Sales distribution (log transformed)
        sns.histplot(np.log(1 + self.df['Item_Outlet_Sales']), ax=axes[1,1], kde=True)
        axes[1,1].set_title('Log Sales Distribution')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model performance comparison
        model_names = list(self.results.keys())
        rmse_scores = [self.results[name]['rmse'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, rmse_scores, color=sns.color_palette("husl", len(model_names)))
        plt.title('Model Performance Comparison (RMSE)', fontsize=16)
        plt.ylabel('RMSE Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance (for tree-based models)
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        best_model = self.results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
            plt.title(f'Top 15 Feature Importance ({best_model_name})', fontsize=16)
            plt.tight_layout()
            plt.savefig(results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Visualizations saved to results directory.")
    
    def save_results(self) -> None:
        """
        Save model results and performance metrics.
        """
        logger.info("Saving results...")
        
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        # Save performance metrics
        performance_df = pd.DataFrame([
            {
                'Model': name,
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2']
            }
            for name, results in self.results.items()
        ])
        
        performance_df.to_csv(results_dir / 'model_performance.csv', index=False)
        
        # Save best model
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        best_model = self.results[best_model_name]['model']
        
        joblib.dump(best_model, results_dir / 'best_model.pkl')
        
        logger.info(f"Results saved. Best model: {best_model_name}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete BigMart sales analysis...")
        
        # Load data
        self.load_data()
        
        # Explore data
        eda_results = self.explore_data()
        
        # Preprocess data
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_categorical_features()
        
        # Prepare features
        X, y = self.prepare_features()
        self.split_data(X, y)
        
        # Train models
        model_results = self.train_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        logger.info("Analysis completed successfully!")
        
        return {
            'eda_results': eda_results,
            'model_results': model_results,
            'best_model': min(model_results.keys(), key=lambda x: model_results[x]['rmse'])
        }


def main():
    """
    Main function to run the BigMart sales analysis.
    """
    parser = argparse.ArgumentParser(description='BigMart Sales Prediction Analysis')
    parser.add_argument('--data-url', type=str, 
                       default='https://raw.githubusercontent.com/GowsalyaKN/BigMart-Sales-Prediction/main/data/Train.csv',
                       help='URL to the training dataset')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = BigMartSalesAnalyzer(data_url=args.data_url)
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Print summary
        print("\n" + "="*50)
        print("BIGMART SALES PREDICTION ANALYSIS SUMMARY")
        print("="*50)
        print(f"Dataset shape: {analyzer.df.shape}")
        print(f"Best model: {results['best_model']}")
        print(f"Best RMSE: {analyzer.results[results['best_model']]['rmse']:.4f}")
        print(f"Best R² Score: {analyzer.results[results['best_model']]['r2']:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 