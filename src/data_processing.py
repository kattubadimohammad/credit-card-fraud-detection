from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """Load the dataset from the given file path."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(df):
    """Clean the dataset without changing the feature scale.

    Random Forest is tree-based, so Amount does not need standardization.
    Keeping Amount in its original units also makes training consistent with
    the Streamlit application's user input.
    """
    try:
        required_columns = {"Class", "Amount"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        df = df.copy()

        if "Time" in df.columns:
            df = df.drop(columns="Time")

        df = df.dropna()
        print("Data preprocessing completed.")
        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


def split_data(df):
    """Split data into stratified train and test sets."""
    try:
        X = df.drop("Class", axis=1)
        y = df["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        print("Data split into stratified train and test sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return None, None, None, None


def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE on the training set only."""
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print("Class imbalance handled using SMOTE.")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error during class imbalance handling: {e}")
        return None, None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    file_path = project_root / "data" / "creditcard.csv"

    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)
        if df is not None:
            X_train, X_test, y_train, y_test = split_data(df)
            if X_train is not None and y_train is not None:
                X_resampled, y_resampled = handle_class_imbalance(X_train, y_train)
                if X_resampled is not None:
                    print("Data preprocessing completed successfully.")
