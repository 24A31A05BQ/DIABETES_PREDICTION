# Diabetes Prediction Using Machine Learning

An educational machine learning application that predicts diabetes outcomes using health measurements and compares multiple classification algorithms.

## Project Overview

The Diabetes Prediction System evaluates four machine learning algorithms across five model configurations. It includes data preprocessing, missing-value handling, cross-validation, hyperparameter tuning, model comparison, and an interactive Streamlit application.

## Dataset Information

- **Dataset:** Pima Indians Diabetes Dataset
- **Total Records:** 768
- **Input Features:** 8
- **Target Column:** Outcome
- **Classes:** 0 (No Diabetes), 1 (Diabetes)

The dataset contains the following input features:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## Machine Learning Algorithms

Four unique algorithms are evaluated across five model configurations:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Tuned Random Forest (GridSearchCV)

## Methodology

- Dataset loading and exploration
- Handling invalid zero values in selected features
- Median imputation for missing values
- Feature scaling where required
- Stratified 80:20 train-test split
- Baseline model training
- Five-fold cross-validation
- Random Forest hyperparameter tuning
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Model comparison and final model selection
- Saving the trained model
- Streamlit application development

## Experimental Results

SVM was selected based on the highest mean five-fold cross-validation F1-score.

| Metric | SVM Result |
|---|---:|
| Accuracy | 72.73% |
| Precision | 59.09% |
| Recall | 72.22% |
| F1-score | 65.00% |
| ROC-AUC | 81.39% |
| Mean CV F1-score | 0.6902 |

Decision Tree achieved the highest test-set F1-score of 70.09%. However, SVM was selected using cross-validation performance rather than test-set performance alone.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## Project Structure

```text
DIABETES_PREDICTION/
|-- app.py
|-- train.py
|-- README.md
|-- requirements.txt
|
|-- dataset/
|   |-- diabetes.csv
|
|-- models/
|   |-- best_model.pkl
|   |-- final_model_name.txt
|
|-- results/
    |-- confusion_matrix.png
    |-- cross_validation.csv
    |-- cross_validation_comparison.png
    |-- model_comparison.csv
    |-- model_comparison.png
    |-- roc_curve.png
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/24A31A05BQ/DIABETES_PREDICTION.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

### 4. Train the Models Again (Optional)

```bash
python train.py
```

The training script generates the trained model and evaluation results.

## Project Contributions

- Implemented missing-value handling for selected health measurements.
- Compared four classification algorithms across five configurations.
- Applied five-fold cross-validation for model selection.
- Performed Random Forest hyperparameter tuning using GridSearchCV.
- Developed an interactive Streamlit application with prediction and analytics pages.

## Limitations

The dataset contains 768 records and represents a specific population. Model performance may not generalize to other populations or clinical settings. Independent validation is necessary before considering any real-world application.

## Disclaimer

This project is intended only for educational and research purposes. It is not a medical diagnostic tool and must not be used to make healthcare decisions. Always consult a qualified healthcare professional for medical advice.
