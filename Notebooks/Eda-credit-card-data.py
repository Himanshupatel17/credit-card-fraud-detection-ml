import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Database connection
engine = create_engine("mysql+pymysql://root:Himanshu%4012@127.0.0.1:3306/fraud_db")

# Load data back into Pandas
df = pd.read_sql("SELECT * FROM transaction", engine)

# Quick check
print(df.shape)
print(df['Class'].value_counts())




#missing values
print(df.isnull().sum())

#eda steps

print(df.shape)
print(df.shape)

print(df.describe)
print(df.head())

print("Missing values per column:\n", df.isnull().sum())

# Class distribution
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), cmap="coolwarm", cbar=False)
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optional SMOTE
# from imblearn.over_sampling import SMOTE

#(Optional) SMOTE for Imbalanced Data
# =====================================
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# print(" After SMOTE:", X_train.shape, y_train.shape)

# Fraud Detection - Model Training


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import  accuracy_score



# Logistic Regression

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("\n🔹 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))



# Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n🔹 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))



# XGBoost Classifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n🔹 XGBoost Results")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))



X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train, columns=["Class"])
y_test_df = pd.DataFrame(y_test, columns=["Class"])

# Save to processed folder
X_train_df.to_csv("D:\domain projects\credit card\dataset\Processed/X_train.csv", index=False)
X_test_df.to_csv("D:\domain projects\credit card\dataset\Processed/X_test.csv", index=False)
y_train_df.to_csv("D:\domain projects\credit card\dataset\Processed/y_train.csv", index=False)
y_test_df.to_csv("D:\domain projects\credit card\dataset\Processed/y_test.csv", index=False)

print(" Preprocessed train/test datasets saved ")


