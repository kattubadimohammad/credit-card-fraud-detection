import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data_preprocessing import load_data, preprocess_data, split_data

def load_model(filename='random_forest_model.pkl'):
    """Load the trained model using Pickle."""
    with open(f'../models/{filename}', 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from ../models/{filename}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using classification report, confusion matrix, and ROC AUC score."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred)}")

def main():
    file_path = '../data/creditcard.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    _, X_test, _, y_test = split_data(df)

    model = load_model()
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
