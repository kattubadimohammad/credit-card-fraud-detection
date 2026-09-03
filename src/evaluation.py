from pathlib import Path
import pickle

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data_processing import load_data, preprocess_data, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"


def load_model(filename=None):
    """Load the trained model from the project's models directory."""
    model_path = Path(filename) if filename else MODEL_PATH
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate with classification, ROC-AUC and PR-AUC metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_proba):.4f}")


def main():
    df = load_data(DATA_PATH)
    if df is None:
        raise RuntimeError("Unable to load the dataset.")

    df = preprocess_data(df)
    if df is None:
        raise RuntimeError("Unable to preprocess the dataset.")

    _, X_test, _, y_test = split_data(df)
    if X_test is None or y_test is None:
        raise RuntimeError("Unable to create the test set.")

    model = load_model()
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
