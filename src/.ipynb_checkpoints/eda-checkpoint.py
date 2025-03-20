import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the dataset for EDA."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_class_distribution(df):
    """Plot the distribution of classes (fraudulent vs. non-fraudulent)."""
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.show()

def plot_transaction_amount(df):
    """Plot transaction amount distribution by class."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Transaction Amount by Class')
    plt.show()

def correlation_heatmap(df):
    """Plot a correlation heatmap to observe feature relationships."""
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.show()

def run_eda(file_path):
    df = load_data(file_path)
    if df is not None:
        print("Performing Exploratory Data Analysis...")
        print(df.head())
        print(df.info())
        print(df.describe())

        plot_class_distribution(df)
        plot_transaction_amount(df)
        correlation_heatmap(df)

        print("EDA completed successfully.")
    else:
        print("EDA aborted due to data loading failure.")

if __name__ == "__main__":
    run_eda('../data/creditcard.csv')
