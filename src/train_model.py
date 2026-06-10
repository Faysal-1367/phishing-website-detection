import pandas as pd
import joblib
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load Features Dataset
df = pd.read_csv("dataset/features.csv")

# Convert Label
df["label"] = df["label"].map({
    "benign": 0,
    "phishing": 1
})

# Features
X = df.drop("label", axis=1)

# Target
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nRESULTS")
print("=" * 30)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# Save Model
joblib.dump(
    model,
    "models/model.pkl"
)

print("\nModel Saved Successfully!")

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.savefig("confusion_matrix.png")
plt.show()