# Credit Card Fraud Detection

A machine-learning project that detects potentially fraudulent credit-card transactions using a Random Forest classifier and SMOTE for the imbalanced training data. A Streamlit interface is included for interactive predictions.

## 🚀 Live Demo

[Credit Card Fraud Detection App](https://credit-card-fraud-detection-0bm5.onrender.com/)

## 📁 Project Structure

```text
credit-card-fraud-detection/
├── app.py
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── data_processing.py
│   ├── eda.py
│   ├── evaluation.py
│   └── model_training.py
├── utils/
├── requirements.txt
└── README.md
```

## 🛠️ Installation

```bash
git clone https://github.com/kattubadimohammad/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

The dataset contains anonymized PCA features (`V1`–`V28`), transaction `Amount`, `Time`, and the binary `Class` target.

Place the extracted `creditcard.csv` file in:

```text
data/creditcard.csv
```

If using the repository's Git LFS dataset pointer, make sure Git LFS is installed and pull the tracked file before running the pipeline.

## ▶️ Usage

### 1. Exploratory Data Analysis

```bash
python src/eda.py
```

### 2. Train the model

```bash
python src/model_training.py
```

Training performs a stratified train/test split and applies SMOTE only to the training set. The trained model is saved to `models/random_forest_model.pkl`.

### 3. Evaluate the model

```bash
python src/evaluation.py
```

Evaluation reports the confusion matrix, classification report, accuracy, ROC-AUC, and PR-AUC (Average Precision).

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## ⚠️ Important: Retrain After the Pipeline Fix

The preprocessing pipeline was updated so that `Amount` remains in its original units. This keeps training and Streamlit inference consistent and removes the previous preprocessing mismatch.

**Retrain the model before relying on the deployed application:**

```bash
python src/model_training.py
```

Then redeploy/restart the Streamlit service so it uses the newly generated model artifact.

## 🧠 Model Notes

- **Algorithm:** Random Forest Classifier
- **Imbalance handling:** SMOTE on training data only
- **Features:** V1–V28 anonymized PCA components + Amount
- **Target:** Class (`0` = legitimate, `1` = fraud)
- **Evaluation:** Precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix

## 📄 License

This project is licensed under the MIT License.
