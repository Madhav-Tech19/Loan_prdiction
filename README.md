# Loan Prediction Project

This repository contains a Jupyter Notebook for predicting loan eligibility using various machine learning techniques. The project demonstrates data preprocessing, feature engineering, model training, and evaluation steps to solve the loan prediction problem effectively.

## Problem Statement
The goal of this project is to build a machine learning model to predict whether a loan applicant is eligible for approval based on specific features such as income, credit history, and loan amount.

## Features
- **Data Preprocessing**: Handling missing values, scaling, and encoding categorical variables.
- **Exploratory Data Analysis**: Visualizing data trends and distributions.
- **Model Training and Evaluation**:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Ensemble Models (e.g., Random Forest)
  - Performance metrics like accuracy, precision, recall, and F1-score.

## Dependencies
This project uses the following Python libraries:
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn`
- **Others**: `warnings`

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/loan-prediction.git
   ```
2. Navigate to the repository:
   ```bash
   cd loan-prediction
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook loan_prediction.ipynb
   ```
4. Follow the notebook to preprocess the dataset, train models, and evaluate their performance.

## Dataset
The dataset used in this project includes features such as:
- Applicant Income
- Loan Amount
- Credit History
- Gender, Marital Status, etc.

Ensure the dataset is in the appropriate directory and modify the file path in the notebook if needed.

## Results
The notebook evaluates the trained models using:
- Confusion Matrix
- Accuracy, Precision, Recall, and F1-Score
- Cross-validation for robust performance assessment

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
