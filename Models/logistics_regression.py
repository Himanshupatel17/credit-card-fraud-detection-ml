

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load preprocessed data
X_train = pd.read_csv("D:\domain projects\credit card\dataset\Processed\X_train.csv").values
X_test = pd.read_csv("D:\domain projects\credit card\dataset\Processed\X_test.csv").values
y_train = pd.read_csv("D:\domain projects\credit card\dataset\Processed\y_train.csv")["Class"].values
y_test = pd.read_csv("D:\domain projects\credit card\dataset\Processed\y_test.csv")["Class"].values

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("../models/logistic_reg.pkl", "wb") as f:
    pickle.dump(model, f)
print(" Logistic Regression model saved successfully.")
