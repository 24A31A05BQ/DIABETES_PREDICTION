
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes AI | Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
MODEL_NAME_PATH = BASE_DIR / "models" / "final_model_name.txt"
RESULTS_DIR = BASE_DIR / "results"
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

ALGORITHMS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine (SVM)",
    "Tuned Random Forest"
]

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #10233F;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #142B4A;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748B;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .info-card {
        background: #FFFFFF;
        padding: 19px;
        border-radius: 13px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .info-label {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
    }

    .info-value {
        color: #142B4A;
        font-size: 27px;
        font-weight: 800;
        margin-top: 5px;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 12px;
        padding: 22px 0 8px 0;
    }

    div.stButton > button {
        background-color: #087F8C;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button:hover {
        background-color: #066873;
        color: white;
    }

    h1, h2, h3 {
        color: #142B4A;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND MODEL NAME
# --------------------------------------------------

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()


def get_final_model_name():
    if MODEL_NAME_PATH.exists():
        try:
            return MODEL_NAME_PATH.read_text(
                encoding="utf-8"
            ).strip()
        except Exception:
            pass

    return "SVM"


final_model_name = get_final_model_name()

# --------------------------------------------------
# COMMON FUNCTIONS
# --------------------------------------------------

def show_header(title, subtitle):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def show_card(label, value):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def load_csv(filename):
    path = RESULTS_DIR / filename

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    return None


def show_result_image(filename, caption):
    path = RESULTS_DIR / filename

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            width="stretch"
        )
    else:
        st.info(
            f"{filename} is not available. Run python train.py first."
        )


def format_metrics_table(data):
    display_data = data.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "Mean CV F1",
        "CV Std"
    ]:
        if column in display_data.columns:
            display_data[column] = display_data[column].round(4)

    return display_data


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")
    st.caption("MACHINE LEARNING PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Diabetes Prediction",
            "Model Analytics",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**Project Information**")
    st.caption("Dataset: Pima Indians Diabetes")
    st.caption("Unique Algorithms: 4")
    st.caption("Model Configurations: 5")
    st.caption(f"Selected Model: {final_model_name}")
    st.caption("Framework: Streamlit")

    st.divider()
    st.caption("Educational ML Project | 2026")

# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if model is None:
    st.error(
        "Trained model not found. Please run python train.py "
        "to generate models/best_model.pkl."
    )
    st.stop()

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    show_header(
        "Diabetes Prediction Dashboard",
        "An overview of the machine learning system and its analytical capabilities."
    )

    st.markdown("### Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_card("DATASET RECORDS", "768")

    with col2:
        show_card("INPUT FEATURES", "08")

    with col3:
        show_card("MODEL CONFIGURATIONS", "05")

    with col4:
        show_card("PREDICTION TYPE", "Binary")

    st.write("")

    comparison = load_csv("model_comparison.csv")
    cross_validation = load_csv("cross_validation.csv")

    left, right = st.columns([1.35, 1])

    with left:

        st.markdown("### About the System")

        st.write(
            "The Diabetes Prediction System is an educational "
            "machine learning application that analyzes eight "
            "health measurements and generates a classification."
        )

        st.write(
            "Four unique classification algorithms are evaluated "
            "across five model configurations using multiple "
            "performance measures and 5-fold cross-validation."
        )

        st.info(
            "Navigate to Diabetes Prediction to enter measurements "
            "and generate an educational model prediction."
        )

    with right:

        st.markdown("### Algorithms Evaluated")

        st.markdown("""
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. Support Vector Machine (SVM)
        5. Tuned Random Forest
        """)

    st.markdown("### Selected Model")

    selected_col1, selected_col2 = st.columns(2)

    with selected_col1:
        show_card("FINAL SELECTED MODEL", final_model_name)

    with selected_col2:

        cv_score = None

        if cross_validation is not None:
            if (
                "Model" in cross_validation.columns
                and "Mean CV F1" in cross_validation.columns
            ):
                selected_row = cross_validation[
                    cross_validation["Model"] == final_model_name
                ]

                if not selected_row.empty:
                    cv_score = float(
                        selected_row.iloc[0]["Mean CV F1"]
                    )

        show_card(
            "MEAN CROSS-VALIDATION F1",
            f"{cv_score:.4f}" if cv_score is not None else "N/A"
        )

    st.markdown("### Model Performance Comparison")

    if comparison is not None:
        st.dataframe(
            format_metrics_table(comparison),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("Run python train.py to generate model results.")

    show_result_image(
        "model_comparison.png",
        "Comparison of the five model configurations"
    )

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Diabetes Prediction":

    show_header(
        "Diabetes Prediction",
        "Enter the required measurements to generate a machine learning prediction."
    )

    st.markdown("### Patient Measurements")

    st.caption(
        "Enter all eight input values. Default values are examples, "
        "not actual patient measurements."
    )

    with st.form("prediction_form"):

        st.markdown("#### Personal and Health Measurements")

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=1.0,
                max_value=300.0,
                value=120.0,
                step=1.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=1.0,
                max_value=200.0,
                value=72.0,
                step=1.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin Level (mu U/ml)",
                min_value=1.0,
                max_value=1000.0,
                value=80.0,
                step=1.0
            )

            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=1.0,
                max_value=100.0,
                value=25.0,
                step=0.1
            )

            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01
            )

            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=30,
                step=1
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "Generate Prediction",
            width="stretch"
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                pedigree,
                age
            ]],
            columns=FEATURES
        )

        try:

            prediction = int(model.predict(input_data)[0])

            st.markdown("### Prediction Result")

            if prediction == 1:
                st.error("Model Prediction: Diabetes")
            else:
                st.success("Model Prediction: No Diabetes")

            st.caption(f"Model used: {final_model_name}")

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)

                if 1 in classes:

                    probability = float(
                        probabilities[classes.index(1)]
                    )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        show_card(
                            "MODEL-ESTIMATED POSITIVE-CLASS SCORE",
                            f"{probability * 100:.2f}%"
                        )

                    with result_col2:
                        show_card(
                            "PREDICTED CLASS",
                            "Diabetes" if prediction == 1
                            else "No Diabetes"
                        )

                    st.progress(probability)

                    st.caption(
                        "This score is the model's estimated probability "
                        "for the positive class. It is not a clinically "
                        "validated individual risk estimate."
                    )

            else:

                st.info(
                    "The selected SVM model does not provide probability "
                    "estimates in its current configuration. "
                    "Only its predicted class is displayed."
                )

            with st.expander("View Submitted Measurements"):

                st.dataframe(
                    input_data,
                    width="stretch",
                    hide_index=True
                )

        except Exception as error:
            st.error(f"Prediction could not be generated: {error}")

    st.warning(
        "Medical Disclaimer: This application is for educational and "
        "research purposes only. It is not a medical diagnostic tool. "
        "Do not use its predictions to make healthcare decisions. "
        "Consult a qualified healthcare professional for medical advice."
    )

# --------------------------------------------------
# MODEL ANALYTICS PAGE
# --------------------------------------------------

elif page == "Model Analytics":

    show_header(
        "Model Analytics",
        "Detailed evaluation and comparison of the five model configurations."
    )

    comparison = load_csv("model_comparison.csv")
    cross_validation = load_csv("cross_validation.csv")

    st.markdown("### Model Performance Comparison")

    if comparison is not None:

        st.dataframe(
            format_metrics_table(comparison),
            width="stretch",
            hide_index=True
        )

        st.markdown("### Performance Visualization")

        show_result_image(
            "model_comparison.png",
            "Accuracy, precision, recall, F1-score and ROC-AUC comparison"
        )

    else:

        st.warning(
            "Model comparison file not found. Run python train.py first."
        )

    st.divider()

    st.markdown("### Cross-Validation Results")

    if cross_validation is not None:

        st.dataframe(
            format_metrics_table(cross_validation),
            width="stretch",
            hide_index=True
        )

        show_result_image(
            "cross_validation_comparison.png",
            "Mean 5-fold cross-validation F1-score comparison"
        )

    else:

        st.info("Cross-validation results are not available.")

    st.divider()

    st.markdown("### Confusion Matrix")

    show_result_image(
        "confusion_matrix.png",
        "Confusion matrix of the selected model"
    )

    st.markdown("### ROC Curve")

    show_result_image(
        "roc_curve.png",
        "ROC curves for the evaluated models"
    )

    st.divider()

    st.markdown("### Feature Importance")

    feature_importance = load_csv("feature_importance.csv")

    if feature_importance is not None:

        st.dataframe(
            feature_importance,
            width="stretch",
            hide_index=True
        )

        show_result_image(
            "feature_importance.png",
            "Feature importance visualization"
        )

    else:

        st.info(
            "Feature importance is not directly available for the "
            "selected SVM model in this experiment."
        )

    st.caption(
        "Evaluation results are experimental and depend on the dataset, "
        "preprocessing, train-test split and model configuration."
    )

# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------

elif page == "About Project":

    show_header(
        "About the Project",
        "Understanding the dataset, algorithms, methodology and technologies."
    )

    # PROJECT DESCRIPTION

    st.markdown("### Project Description")

    st.write(
        "The Diabetes Prediction System is an educational machine "
        "learning project that evaluates four classification algorithms "
        "across five model configurations to predict diabetes outcomes "
        "using eight input measurements."
    )

    # PROJECT OBJECTIVES

    st.markdown("### Project Objectives")

    st.markdown("""
    - Develop a machine learning-based prediction system.
    - Handle missing and invalid zero values in selected features.
    - Compare multiple classification algorithms.
    - Perform hyperparameter tuning.
    - Evaluate models using multiple performance metrics.
    - Apply cross-validation for model selection.
    - Develop an interactive application using Streamlit.
    """)

    # DATASET INFORMATION

    st.markdown("### Dataset Information")

    st.write("**Dataset:** Pima Indians Diabetes Dataset")

    st.markdown("""
    - Total records: 768
    - Input features: 8
    - Target column: Outcome
    - Prediction classes: 0 (No Diabetes), 1 (Diabetes)
    """)

    if DATASET_PATH.exists():

        try:

            dataset = pd.read_csv(DATASET_PATH)

            st.markdown("#### Dataset Preview")

            st.dataframe(
                dataset.head(5),
                width="stretch"
            )

        except Exception:

            st.info("Dataset preview could not be loaded.")

    # MACHINE LEARNING ALGORITHMS

    st.markdown("### Machine Learning Algorithms")

    st.caption(
        "Four unique algorithms are evaluated across five model configurations."
    )

    algorithms = {
        "Logistic Regression":
            "A linear classification algorithm used as a baseline model.",

        "Decision Tree":
            "A tree-based algorithm that makes predictions using decision rules.",

        "Random Forest":
            "An ensemble algorithm that combines multiple decision trees.",

        "Support Vector Machine (SVM)":
            "A classification algorithm that identifies a separating decision boundary.",

        "Tuned Random Forest":
            "A Random Forest model whose hyperparameters are optimized using GridSearchCV."
    }

    for name, description in algorithms.items():

        with st.expander(name):
            st.write(description)

    # MODEL SELECTION

    st.markdown("### Model Selection")

    st.success(f"Selected Model: {final_model_name}")

    st.write(
        "Support Vector Machine (SVM) was selected based on its highest "
        "mean 5-fold cross-validation F1-score of 0.6902 among the "
        "evaluated model configurations."
    )

    st.markdown("#### Selected Model Test Performance")

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
        st.columns(5)
    )

    with metric_col1:
        show_card("ACCURACY", "72.73%")

    with metric_col2:
        show_card("PRECISION", "59.09%")

    with metric_col3:
        show_card("RECALL", "72.22%")

    with metric_col4:
        show_card("F1-SCORE", "65.00%")

    with metric_col5:
        show_card("ROC AUC", "81.39%")

    st.info(
        "Decision Tree achieved the highest test-set F1-score of 70.09%. "
        "SVM was selected using cross-validation performance rather "
        "than choosing a model solely from the test-set result."
    )

    # TECHNOLOGIES USED

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib and Seaborn
    - Joblib
    - Streamlit
    """)

    # PROJECT WORKFLOW

    st.markdown("### Project Workflow")

    st.markdown("""
    1. Dataset collection
    2. Data preprocessing
    3. Missing-value handling
    4. Feature scaling
    5. Train-test splitting
    6. Baseline model training
    7. Cross-validation
    8. Random Forest hyperparameter tuning
    9. Model evaluation
    10. Model comparison
    11. Final model selection and saving
    12. Streamlit application development
    """)

    # PROJECT CONTRIBUTIONS

    st.markdown("### Project Contributions")

    st.markdown("""
    1. Handled invalid zero values in selected medical measurements
       using missing-value imputation.
    2. Compared four machine learning algorithms across five
       model configurations.
    3. Used 5-fold cross-validation for model evaluation and selection.
    4. Applied GridSearchCV-based hyperparameter tuning to Random Forest.
    5. Developed an interactive Streamlit application for predictions
       and model analytics.
    """)

    # EXPERIMENTAL OBSERVATION

    st.markdown("### Experimental Observation")

    st.info(
        "SVM achieved the highest mean cross-validation F1-score (0.6902). "
        "Decision Tree achieved the highest test-set F1-score (0.7009). "
        "These are different evaluation measures, so both results are "
        "reported separately."
    )

    # DATASET LIMITATIONS

    st.markdown("### Dataset Limitations")

    st.write(
        "The dataset contains 768 records and represents a specific "
        "population. Results may not generalize to other populations "
        "or clinical settings. Independent validation is required "
        "before any real-world medical application."
    )

    # DISCLAIMER

    st.markdown("### Disclaimer")

    st.warning(
        "This project is intended only for educational and research "
        "purposes. It is not a medical diagnostic system and must not "
        "replace professional medical advice."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Diabetes AI Prediction System | Machine Learning Project<br>
    Developed for Educational and Research Purposes
</div>
""", unsafe_allow_html=True)
