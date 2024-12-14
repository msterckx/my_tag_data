import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

def main():
    # Process easy dataset
    df = pd.read_csv('data/easy_dataset.csv')
    
    # Create correlation plot
    create_correlation_plot(df)
    
    # Create distribution plots
    create_distribution_plots(df)

if __name__ == "__main__":
    main()
