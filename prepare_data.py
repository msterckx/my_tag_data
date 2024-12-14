import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_correlation_plot(df):
    """Create and save correlation heatmap for the given dataframe."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations - Easy Dataset')
    
    # Ensure images directory exists
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Save plot
    output_path = os.path.join('images', 'correlation_easy_dataset.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_distribution_plots(df):
    """Create and save distribution plots for each feature."""
    # Ensure images directory exists
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Create distribution plot for each numerical column
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        if column != 'target':  # Skip target column
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df, x=column, kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            
            # Save plot
            output_path = os.path.join('images', f'distribution_{column}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

def train_and_evaluate_model(df):
    """Train a CatBoost model and evaluate its accuracy."""
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=100,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Ensure directory exists
    os.makedirs('/kaggle/working', exist_ok=True)
    
    # Save the model
    model_path = '/kaggle/working/catboost_model.cbm'
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, accuracy

def main():
    # Read dataset
    df = pd.read_csv('data/easy_dataset.csv')
    
    # Create correlation plot
    create_correlation_plot(df)
    
    # Create distribution plots
    create_distribution_plots(df)
    
    # Train and evaluate model
    model, accuracy = train_and_evaluate_model(df)

if __name__ == "__main__":
    main()
