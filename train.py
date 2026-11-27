
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. CREATE PROJECT FOLDERS
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 65)
print(" DIABETES PREDICTION USING MACHINE LEARNING")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

data = pd.read_csv("dataset/diabetes.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

if "Outcome" not in data.columns:
    raise ValueError("Dataset must contain an Outcome column.")

# --------------------------------------------------
# 3. DATA PREPROCESSING
# --------------------------------------------------

invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[invalid_zero_columns] = data[
    invalid_zero_columns
].replace(0, np.nan)

print("\nMissing values after zero-value handling:")
print(data.isnull().sum())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. DEFINE BASELINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced"
        ))
    ])
}

# --------------------------------------------------
# 6. DEFINE TUNED RANDOM FOREST
# --------------------------------------------------

tuned_rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

parameter_grid = {
    "classifier__n_estimators": [150, 250],
    "classifier__max_depth": [None, 8],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [2, 5]
}

# --------------------------------------------------
# 7. CROSS-VALIDATION OF BASELINE MODELS
# --------------------------------------------------

print("\n" + "=" * 65)
print(" BASELINE MODEL CROSS-VALIDATION")
print("=" * 65)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, model in models.items():

    print("\nEvaluating:", name)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    mean_score = scores.mean()
    std_score = scores.std()

    cv_results.append({
        "Model": name,
        "Mean CV F1": mean_score,
        "CV Std": std_score
    })

    print("Fold scores:", np.round(scores, 4))
    print("Mean CV F1:", round(mean_score, 4))
    print("CV Standard Deviation:", round(std_score, 4))

# --------------------------------------------------
# 8. NESTED CROSS-VALIDATION FOR TUNED RANDOM FOREST
# --------------------------------------------------

print("\n" + "=" * 65)
print(" TUNED RANDOM FOREST CROSS-VALIDATION")
print("=" * 65)

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

rf_search = GridSearchCV(
    estimator=tuned_rf_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=inner_cv,
    n_jobs=-1,
    refit=True
)

# Outer CV evaluates the complete tuning procedure.
# This gives a fairer estimate than reporting the best
# score from the same grid search used for selection.

nested_scores = cross_val_score(
    rf_search,
    X_train,
    y_train,
    cv=cv,
    scoring="f1",
    n_jobs=1
)

tuned_cv_mean = nested_scores.mean()
tuned_cv_std = nested_scores.std()

cv_results.append({
    "Model": "Tuned Random Forest",
    "Mean CV F1": tuned_cv_mean,
    "CV Std": tuned_cv_std
})

print("Nested CV fold scores:", np.round(nested_scores, 4))
print("Mean Nested CV F1:", round(tuned_cv_mean, 4))
print("Nested CV Standard Deviation:", round(tuned_cv_std, 4))

# --------------------------------------------------
# 9. SAVE CROSS-VALIDATION RESULTS
# --------------------------------------------------

cv_comparison = pd.DataFrame(cv_results)

cv_comparison = cv_comparison.sort_values(
    by="Mean CV F1",
    ascending=False
)

cv_comparison.to_csv(
    "results/cross_validation.csv",
    index=False
)

print("\nCROSS-VALIDATION COMPARISON")
print(cv_comparison.to_string(index=False))

# --------------------------------------------------
# 10. FIT BASELINE MODELS
# --------------------------------------------------

print("\n" + "=" * 65)
print(" TRAINING BASELINE MODELS")
print("=" * 65)

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)
    trained_models[name] = model

# --------------------------------------------------
# 11. TUNE AND TRAIN RANDOM FOREST
# --------------------------------------------------

print("\n" + "=" * 65)
print(" FINAL RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 65)

final_rf_search = GridSearchCV(
    estimator=tuned_rf_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=inner_cv,
    n_jobs=-1,
    refit=True
)

final_rf_search.fit(X_train, y_train)

tuned_rf_model = final_rf_search.best_estimator_

trained_models["Tuned Random Forest"] = tuned_rf_model

print("\nBest Random Forest Parameters:")
print(final_rf_search.best_params_)

print(
    "\nBest inner GridSearchCV F1:",
    round(final_rf_search.best_score_, 4)
)

# --------------------------------------------------
# 12. EVALUATE ALL MODELS ON TEST DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print(" TEST SET MODEL EVALUATION")
print("=" * 65)

test_results = []
test_predictions = {}

for name, model in trained_models.items():

    print("\nEvaluating:", name)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(y_test, probabilities)

    matching_cv = cv_comparison[
        cv_comparison["Model"] == name
    ].iloc[0]

    test_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Mean CV F1": matching_cv["Mean CV F1"],
        "CV Std": matching_cv["CV Std"]
    })

    test_predictions[name] = {
        "predictions": predictions,
        "probabilities": probabilities
    }

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))
    print("Mean CV F1:", round(matching_cv["Mean CV F1"], 4))

# --------------------------------------------------
# 13. FINAL MODEL SELECTION
# --------------------------------------------------

# Select using training-set cross-validation results,
# NOT by selecting the highest test-set score.

best_name = cv_comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\n" + "=" * 65)
print(" FINAL MODEL SELECTION")
print("=" * 65)

print("\nSelected Model:", best_name)

print(
    "Mean Cross-Validation F1:",
    round(cv_comparison.iloc[0]["Mean CV F1"], 4)
)

# --------------------------------------------------
# 14. SAVE COMPARISON RESULTS
# --------------------------------------------------

comparison = pd.DataFrame(test_results)

comparison = comparison.sort_values(
    by="Mean CV F1",
    ascending=False
)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# --------------------------------------------------
# 15. SAVE FINAL MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nFinal model saved successfully!")

# Save selected model name for project documentation
with open("models/final_model_name.txt", "w") as file:
    file.write(best_name)

# --------------------------------------------------
# 16. MODEL COMPARISON GRAPH
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

plt.figure(figsize=(13, 7))

comparison.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(13, 7)
)

plt.title(
    "Performance Comparison of Machine Learning Models",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel("Performance Score")
plt.xlabel("Machine Learning Algorithms")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 17. CROSS-VALIDATION GRAPH
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=cv_comparison,
    x="Model",
    y="Mean CV F1"
)

plt.title("Cross-Validation F1 Score Comparison")
plt.ylabel("Mean 5-Fold / Nested CV F1")
plt.xlabel("Machine Learning Models")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/cross_validation_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 18. CONFUSION MATRIX
# --------------------------------------------------

best_predictions = test_predictions[best_name]["predictions"]

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 19. ROC CURVE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name, result in test_predictions.items():

    probabilities = result["probabilities"]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison of All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 20. FEATURE IMPORTANCE
# --------------------------------------------------

classifier = best_model.named_steps["classifier"]

if hasattr(classifier, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": classifier.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Feature Importance - " + best_name)
    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300
    )

    plt.close()

else:
    print(
        "\nFeature importance is not directly available "
        "for the selected model."
    )

    if os.path.exists("results/feature_importance.csv"):
        os.remove("results/feature_importance.csv")

    if os.path.exists("results/feature_importance.png"):
        os.remove("results/feature_importance.png")

# --------------------------------------------------
# 21. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("Selected Model:", best_name)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["No Diabetes", "Diabetes"],
        zero_division=0
    )
)

# --------------------------------------------------
# 22. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print(" PROJECT EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nFinal Selected Model:", best_name)

print("\nGenerated Files:")
print("1. models/best_model.pkl")
print("2. models/final_model_name.txt")
print("3. results/model_comparison.csv")
print("4. results/model_comparison.png")
print("5. results/cross_validation.csv")
print("6. results/cross_validation_comparison.png")
print("7. results/confusion_matrix.png")
print("8. results/roc_curve.png")
print("9. results/feature_importance.csv (if supported)")
print("10. results/feature_importance.png (if supported)")

print("\nAll experiments and evaluations completed!")
