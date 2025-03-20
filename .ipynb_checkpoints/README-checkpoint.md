# Credit Card Fraud Detection Project

## Project Overview
This project aims to build a classification model to detect fraudulent credit card transactions. By analyzing transaction data, the model will predict whether a given transaction is fraudulent or not.

## Project Structure
```
Credit_Card_Fraud_Detection_Project/
│
├── data/
│   ├── creditcard.csv               # Dataset
│
├── notebooks/
│   ├── Credit_Card_Fraud_Detection.ipynb   # Jupyter Notebook with complete code
│
├── src/
│   ├── data_preprocessing.py        # Data cleaning and preprocessing
│   ├── eda.py                       # Exploratory Data Analysis (EDA)
│   ├── model_training.py            # Model training using Random Forest
│   ├── evaluation.py                # Model evaluation
│
├── models/
│   ├── random_forest_model.pkl      # Saved trained model using Pickle
│
├── reports/
│   ├── eda_report.png               # Optional: Visualizations from EDA
│   ├── model_performance.txt        # Evaluation results and metrics
│
├── utils/
│   ├── logger.py                    # Logging setup for debugging
│
├── requirements.txt                 # List of Python packages
├── README.md                        # Project documentation
└── app.py                           # Optional: Streamlit app for visualization
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/username/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection_Project
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate # For Linux/Mac
env\Scripts\activate # For Windows
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preprocessing:**
```bash
python src/data_preprocessing.py
```

2. **Exploratory Data Analysis:**
```bash
python src/eda.py
```

3. **Model Training:**
```bash
python src/model_training.py
```

4. **Evaluation:**
```bash
python src/evaluation.py
```

5. **Run the App (Optional):**
```bash
streamlit run app.py
```

## Results
- Evaluation metrics such as accuracy, precision, recall, and ROC-AUC score will be displayed.
- Visualizations from the EDA process will be available under the `reports/` directory.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, please contact **miraclemohammad786@gmail.com**.
