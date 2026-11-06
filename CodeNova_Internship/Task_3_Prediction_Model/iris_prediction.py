
# CodeNova Internship - Task 3
# Iris Flower Classification and Prediction

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

# 1. Load dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

os.makedirs("outputs", exist_ok=True)

print("IRIS FLOWER CLASSIFICATION")
print("Dataset shape:", X.shape)
print("\nMissing values:", X.isnull().sum().sum())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# 3. Define baseline and ML models
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []
trained_models = {}

# 4. Train, cross-validate and evaluate
for name, model in models.items():

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="accuracy"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro"
    )
    recall = recall_score(
        y_test, predictions, average="macro"
    )
    f1 = f1_score(
        y_test, predictions, average="macro"
    )

    results.append({
        "Model": name,
        "CV Accuracy": cv_scores.mean(),
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print("\nMODEL:", name)
    print("Cross-validation accuracy:", round(cv_scores.mean(), 4))
    print("Test accuracy:", round(accuracy, 4))
    print(classification_report(
        y_test, predictions,
        target_names=iris.target_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"outputs/{filename}_confusion_matrix.png")
    plt.close()

    # Multiclass ROC-AUC where probabilities are available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        y_binary = label_binarize(y_test, classes=[0, 1, 2])

        auc = roc_auc_score(
            y_binary, probabilities,
            average="macro", multi_class="ovr"
        )
        print("ROC-AUC:", round(auc, 4))

# 5. Compare models
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("outputs/model_comparison.csv", index=False)

# Model comparison chart
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Test Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Test Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# 6. Feature importance
rf = trained_models["Random Forest"]

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

print("\nFEATURE IMPORTANCE")
print(importance)

# 7. Select best model using cross-validation
best_name = results_df.loc[
    results_df["CV Accuracy"].idxmax(), "Model"
]

print("\nBEST MODEL:", best_name)

best_model = trained_models[best_name]

# 8. Sample flower prediction
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = best_model.predict(sample)[0]

print("\nSAMPLE FLOWER PREDICTION")
print("Predicted flower:", iris.target_names[prediction])

# 9. Save model
joblib.dump(best_model, "outputs/best_iris_model.pkl")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Check the outputs folder.")
