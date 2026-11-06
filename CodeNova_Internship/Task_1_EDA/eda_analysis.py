
# CODENOVA INTERNSHIP - TASK 1
# Exploratory Data Analysis of Wine Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

# STEP 1: Load Dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()

df.rename(columns={"target": "wine_class"}, inplace=True)

print("========== DATASET INFORMATION ==========")

# STEP 2: Dataset Overview
print("\nDataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

# STEP 3: Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# STEP 4: Duplicate Removal
print("\nDuplicate Records:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# STEP 5: Outlier Detection
numeric = df.select_dtypes(include=np.number).drop(
    columns=["wine_class"]
)

print("\nOutlier Analysis:")

for col in numeric.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[col] < lower) | (df[col] > upper)).sum()

    print(col, ":", count, "potential outliers")

# Create output folder
import os
os.makedirs("outputs", exist_ok=True)

# STEP 6: Correlation Heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), annot=False, cmap="coolwarm")

plt.title("Correlation Heatmap - Wine Dataset")
plt.tight_layout()
plt.savefig("outputs/heatmap.png")
plt.show()

# STEP 7: Feature Distribution
df[numeric.columns].hist(figsize=(14, 10), bins=20)

plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("outputs/distributions.png")
plt.show()

# STEP 8: Boxplot
plt.figure(figsize=(14, 7))

sns.boxplot(data=df[numeric.columns])

plt.xticks(rotation=90)
plt.title("Boxplot - Outlier Detection")
plt.tight_layout()
plt.savefig("outputs/boxplot.png")
plt.show()

# STEP 9: Pairwise Scatter Plot
selected = [
    "alcohol",
    "malic_acid",
    "ash",
    "flavanoids",
    "color_intensity",
    "wine_class"
]

sns.pairplot(df[selected], hue="wine_class")

plt.savefig("outputs/pairplot.png")
plt.show()

# STEP 10: Save Cleaned Dataset
df.to_csv("outputs/cleaned_dataset.csv", index=False)

print("\n========== KEY OBSERVATIONS ==========")

print("1. Dataset contains", df.shape[0], "records.")
print("2. Dataset contains", df.shape[1]-1, "input features.")
print("3. Missing values:", df.isnull().sum().sum())
print("4. Dataset contains three wine classes.")
print("5. Visualizations help identify feature distributions,")
print("   correlations and potential outliers.")

print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
print("Check the outputs folder for graphs and CSV file.")
