# Credit Card Fraud Detection Project

## Deployment
You can access the deployed application using the following link:
[Credit Card Fraud Detection App](https://credit-card-fraud-detection-0bm5.onrender.com/)

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

## Dataset
The dataset used for this project is available on Kaggle:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download the dataset and place the zipped file in the `data/` directory as `creditcard.zip`.
Run the following script to extract it:
```bash
python utils/compress.py
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

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, please contact **miraclemohammad786@gmail.com**.

