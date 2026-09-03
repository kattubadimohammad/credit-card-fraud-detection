from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier

from data_processing import load_data, preprocess_data, split_data, handle_class_imbalance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"


def train_model(X_train, y_train):
    """Train a Random Forest classifier on the resampled training data."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename=None):
    """Save the trained model to the project's models directory."""
    model_path = Path(filename) if filename else MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")


def main():
    df = load_data(DATA_PATH)
    if df is None:
        raise RuntimeError("Unable to load the dataset.")

    df = preprocess_data(df)
    if df is None:
        raise RuntimeError("Unable to preprocess the dataset.")

    X_train, _, y_train, _ = split_data(df)
    if X_train is None or y_train is None:
        raise RuntimeError("Unable to split the dataset.")

    X_resampled, y_resampled = handle_class_imbalance(X_train, y_train)
    if X_resampled is None or y_resampled is None:
        raise RuntimeError("Unable to balance the training data.")

    model = train_model(X_resampled, y_resampled)
    save_model(model)
    print("Model training completed successfully.")


if __name__ == "__main__":
    main()
