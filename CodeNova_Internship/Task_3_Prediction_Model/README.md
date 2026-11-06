# CodeNova Internship – Task 3
## Iris Flower Classification and Prediction

### Project Overview
This project develops a machine learning classification model to predict Iris flower species using the Iris dataset.

### Dataset
- Dataset: Iris Dataset
- Source: Scikit-learn
- Total samples: 150
- Features: 4
- Classes: Setosa, Versicolor, Virginica

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

### Machine Learning Models
1. Dummy Classifier (Baseline)
2. Logistic Regression
3. Random Forest Classifier

### Methodology
1. Dataset loading and inspection
2. Train-test splitting
3. Feature scaling
4. Five-fold cross-validation
5. Model training and comparison
6. Performance evaluation
7. Confusion matrix visualization
8. Feature importance analysis
9. Sample flower prediction

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Outputs
- Model comparison chart
- Confusion matrices
- Feature importance chart
- Model comparison CSV
- Trained machine learning model

### How to Run
Install dependencies:

`pip install pandas numpy matplotlib seaborn scikit-learn joblib`

Run:

`python iris_prediction.py`

### Limitations
- The dataset is relatively small.
- The model is trained on the Iris dataset and may not generalize to other flower species.
- Performance may vary with different train-test splits.

### Conclusion
This project demonstrates the implementation and evaluation of machine learning classification algorithms for predicting Iris flower species.