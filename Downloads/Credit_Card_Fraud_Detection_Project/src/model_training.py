import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
from data_preprocessing import load_data, preprocess_data, split_data, handle_class_imbalance

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='random_forest_model.pkl'):
    """Save the trained model using Pickle."""
    with open(f'models/{filename}', 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to models/{filename}")

def main():
    file_path = 'data/creditcard.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_resampled, y_resampled = handle_class_imbalance(X_train, y_train)

    model = train_model(X_resampled, y_resampled)
    save_model(model)

    print("Model Training Completed.")

if __name__ == "__main__":
    main()
