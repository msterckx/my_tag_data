import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_correlation_plot(df, filename):
    """Create and save correlation heatmap for the given dataframe."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlations - {filename}')
    
    # Ensure images directory exists
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Save plot
    output_path = os.path.join('images', f'correlation_{filename}.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Process easy dataset
    easy_df = pd.read_csv('data/easy_dataset.csv')
    create_correlation_plot(easy_df, 'easy_dataset')
    
    # Process hard dataset
    hard_df = pd.read_csv('data/hard_dataset.csv')
    create_correlation_plot(hard_df, 'hard_dataset')

if __name__ == "__main__":
    main()
