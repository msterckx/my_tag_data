import pandas as pd
import numpy as np
from typing import Tuple, Dict
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
import optuna

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

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cat_features: list) -> float:
    """
    Optuna objective function for hyperparameter optimization
    """
    # Define the hyperparameters to optimize
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'loss_function': 'Logloss',
        'verbose': 0
    }
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create Pool objects
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # Train model
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
        
        # Evaluate
        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)

def train_model(train_df: pd.DataFrame) -> Tuple[CatBoostClassifier, float]:
    """
    Train CatBoost model using Optuna for hyperparameter optimization
    Args:
        train_df: Training dataframe
    Returns:
        Tuple containing trained CatBoost model and validation accuracy
    """
    # Prepare features
    X = train_df.drop(['Transported', 'PassengerId'], axis=1, errors='ignore')
    y = train_df['Transported']
    
    # Identify categorical features
    cat_features = X.select_dtypes(include=['category']).columns.tolist()
    
    # Create Optuna study for hyperparameter optimization
    print("\nOptimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, cat_features), n_trials=20)
    
    # Get best parameters
    best_params = study.best_params
    best_params['loss_function'] = 'Logloss'
    print(f"\nBest parameters found: {best_params}")
    
    # Train final model with best parameters using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    final_model = None
    best_score = 0
    
    print("\nTraining final model with cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create Pool objects
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # Train model
        model = CatBoostClassifier(**best_params)
        model.fit(train_pool, eval_set=val_pool, verbose=100)
        
        # Evaluate
        score = accuracy_score(y_val, model.predict(X_val))
        print(f"Fold {fold} accuracy: {score:.4f}")
        
        # Keep the model with the best validation score
        if score > best_score:
            best_score = score
            final_model = model
    
    print(f"\nBest validation accuracy: {best_score:.4f}")
    return final_model, best_score

def create_submission(model: CatBoostClassifier, test_df: pd.DataFrame):
    """
    Create submission file with model predictions
    Args:
        model: Trained CatBoost model
        test_df: Test dataframe
    """
    # Prepare test features
    X_test = test_df.drop(['PassengerId'], axis=1, errors='ignore')
    
    # Create test pool
    cat_features = X_test.select_dtypes(include=['category']).columns.tolist()
    test_pool = Pool(X_test, cat_features=cat_features)
    
    # Make predictions
    print("\nMaking predictions on test data...")
    predictions = model.predict(test_pool)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predictions
    })
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    submission_path = f'data/{timestamp}_submission.csv'
    
    # Save submission file
    print(f"\nSaving submission to {submission_path}")
    submission_df.to_csv(submission_path, index=False)
    print("Submission file created successfully")

def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, CatBoostClassifier, float]:
    """
    Main function to prepare the data and train model
    Returns:
        Tuple containing processed training data, test data, trained model, and validation accuracy
    """
    # Read the data
    print("Reading data...")
    train_df, test_df = read_data()
    
    # Handle missing values
    print("\nHandling missing values in training data...")
    train_df = handle_missing_values(train_df, is_train=True)
    print("\nHandling missing values in test data...")
    test_df = handle_missing_values(test_df, is_train=False)
    
    # Prepare features
    print("\nPreparing features...")
    train_df = prepare_features(train_df)
    test_df = prepare_features(test_df)
    
    # Train model and get validation accuracy
    model, val_accuracy = train_model(train_df)
    
    return train_df, test_df, model, val_accuracy

if __name__ == "__main__":
    # Execute the data preparation and model training
    train_processed, test_processed, model, val_accuracy = prepare_data()
    
    # Save the trained model
    model_path = 'catboost_model.cbm'
    print(f"\nSaving model to {model_path}")
    model.save_model(model_path)
    
    # Create submission file
    create_submission(model, test_processed)
    
    # Print basic information about the processed datasets
    print("\nProcessed training data info:")
    print(train_processed.info())
    print("\nProcessed test data info:")
    print(test_processed.info())
    print(f"\nFinal model validation accuracy: {val_accuracy:.4f}")
