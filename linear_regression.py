
# TASK 3: LINEAR REGRESSION - HOUSING DATASET


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create Output Folder

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load Dataset

df = pd.read_csv(r"E:\New folder (3)\OneDrive\Desktop\Elevate Labs\Day3\Housing.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


# Data Preprocessing


# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model Training

model = LinearRegression()
model.fit(X_train, y_train)


# Predictions

y_pred = model.predict(X_test)


# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# Plot 1: Actual vs Predicted

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.close()


# Plot 2: Residual Plot

residuals = y_test - y_pred

plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"))
plt.close()


# Plot 3: Correlation Heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
plt.close()

print("Linear Regression Task Completed Successfully.")
