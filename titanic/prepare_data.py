import pandas as pd
import numpy as np
from typing import Tuple
from catboost import CatBoostClassifier
import os

def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read training and test data from CSV files
    Returns:
        Tuple containing training and test dataframes
    """
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    return train_df, test_df

def handle_missing_values(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Handle missing values in the dataframe
    Args:
        df: Input dataframe
        is_train: Boolean indicating if this is training data
    Returns:
        Dataframe with handled missing values
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Get missing value statistics
    missing_stats = df.isnull().sum() / len(df)
    
    # Handle each column based on its type and missing percentage
    for column in df.columns:
        missing_pct = missing_stats[column]
        
        # Skip if no missing values
        if missing_pct == 0:
            continue
            
        # Drop columns with more than 50% missing values
        if missing_pct > 0.5:
            print(f"Dropping column {column} due to {missing_pct:.2%} missing values")
            df = df.drop(columns=[column])
            continue
        
        # For remaining columns, impute based on data type
        if df[column].dtype in ['int64', 'float64']:
            # Use median for numerical columns
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            print(f"Filled missing values in {column} with median: {median_value}")
        else:
            # Use mode for categorical columns
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
            print(f"Filled missing values in {column} with mode: {mode_value}")
    
    return df

def handle_outliers(df: pd.DataFrame, numeric_columns: list = None) -> pd.DataFrame:
    """
    Handle outliers in numerical columns using IQR method
    Args:
        df: Input dataframe
        numeric_columns: List of numerical columns to check for outliers
    Returns:
        Dataframe with handled outliers
    """
    df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    for column in numeric_columns:
        # Skip non-numeric columns
        if df[column].dtype not in ['int64', 'float64']:
            continue
            
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        outliers_low = (df[column] < lower_bound).sum()
        outliers_high = (df[column] > upper_bound).sum()
        
        if outliers_low > 0 or outliers_high > 0:
            print(f"Capping outliers in {column}:")
            print(f"  - Lower bound ({lower_bound:.2f}): {outliers_low} outliers")
            print(f"  - Upper bound ({upper_bound:.2f}): {outliers_high} outliers")
            
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training
    Args:
        df: Input dataframe
    Returns:
        Dataframe with prepared features
    """
    df = df.copy()
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Convert categorical columns to category type for CatBoost
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    return df

def train_model(train_df: pd.DataFrame) -> CatBoostClassifier:
    """
    Train CatBoost model
    Args:
        train_df: Training dataframe
    Returns:
        Trained CatBoost model
    """
    # Prepare features
    X = train_df.drop(['Transported', 'PassengerId'], axis=1, errors='ignore')
    y = train_df['Transported']
    
    # Identify categorical features
    cat_features = X.select_dtypes(include=['category']).columns.tolist()
    
    # Initialize and train model
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        verbose=100
    )
    
    print("\nTraining CatBoost model...")
    model.fit(
        X, y,
        cat_features=cat_features,
        plot=False
    )
    
    return model

def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, CatBoostClassifier]:
    """
    Main function to prepare the data and train model
    Returns:
        Tuple containing processed training data, test data, and trained model
    """
    # Read the data
    print("Reading data...")
    train_df, test_df = read_data()
    
    # Handle missing values
    print("\nHandling missing values in training data...")
    train_df = handle_missing_values(train_df, is_train=True)
    print("\nHandling missing values in test data...")
    test_df = handle_missing_values(test_df, is_train=False)
    
    # Handle outliers
    print("\nHandling outliers in training data...")
    train_df = handle_outliers(train_df)
    print("\nHandling outliers in test data...")
    test_df = handle_outliers(test_df)
    
    # Prepare features
    print("\nPreparing features...")
    train_df = prepare_features(train_df)
    test_df = prepare_features(test_df)
    
    # Train model
    model = train_model(train_df)
    
    return train_df, test_df, model

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('/kaggle/working', exist_ok=True)
    
    # Execute the data preparation and model training
    train_processed, test_processed, model = prepare_data()
    
    # Save the trained model
    model_path = '/kaggle/working/catboost_model.cbm'
    print(f"\nSaving model to {model_path}")
    model.save_model(model_path)
    
    # Print basic information about the processed datasets
    print("\nProcessed training data info:")
    print(train_processed.info())
    print("\nProcessed test data info:")
    print(test_processed.info())
