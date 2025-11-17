# Credit Card Fraud Detection | Machine Learning Project
Detecting fraudulent financial transactions using ML, SQL, Streamlit, and Power BI.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Key Insights](#key-insights)
- [Dashboard / Model / Output](#dashboard--model--output)
- [How to Run This Project](#how-to-run-this-project)
- [Results & Conclusion](#results--conclusion)
- [Future Work](#future-work)
- [Author & Contact](#author--contact)

---

## Overview
This project builds an end-to-end ML system for identifying fraudulent credit card transactions.

The system includes:
- Data preprocessing & SMOTE balancing  
- Training multiple ML models  
- Streamlit deployment  
- MySQL logging  
- Power BI dashboard  

It replicates real-world financial fraud detection workflows used by banks and fintech companies.

---

## Problem Statement
Credit card fraud causes massive financial losses.

Traditional rule-based systems fail because:
- Fraud patterns evolve
- Attackers adapt quickly
- Rules can't detect subtle anomalies

### **Goal**
Build an ML system that predicts fraudulent transactions using PCA-transformed features.

---

## Dataset
- **Source:** Kaggle – Credit Card Fraud Dataset  
- **Rows:** 284,807  
- **Features:**  
  - Time, Amount  
  - V1–V28 (PCA features)  
  - Class (0 = Legit, 1 = Fraud)

**Imbalance:**  
Fraud = only 0.17% → solved using **SMOTE**.

---

## Tools & Technologies
- Python (Pandas, NumPy, Scikit-learn, XGBoost, imblearn)
- Streamlit
- MySQL, PyMySQL, SQLAlchemy
- Power BI
- Git & GitHub

---

## Project Structure

credit-card-fraud-detection-ml/
│
├── app.py # Streamlit app
├── README.md
├── requirements.txt
│
├── data/
│ ├── raw/
│ ├── processed/ # Train/test after SMOTE + scaling
│
├── notebooks/
│ ├── EDA, model training, comparison
│
├── models/ # Saved ML models (.pkl)
│
├── scripts/ # Automation & SQL scripts
│
├── reports/ # Final PDF, PPT, screenshots
│
└── docs/ # Additional documentation

yaml
Copy code

---

## Methods

### **1. Data Preprocessing**
- Standardization  
- SMOTE oversampling  
- Train-test split  

### **2. Model Training**
Trained 3 models:
- Logistic Regression  
- Random Forest (best)  
- XGBoost  

### **3. Deployment**
- Saved best model as `.pkl`
- Streamlit UI:
  - Single input form (30 features)
  - CSV upload
  - Prediction summary
- Auto-store predictions in MySQL  
- Power BI dashboard for analytics  

---

## Key Insights
- Fraud vs non-fraud clearly separable in PCA space  
- SMOTE massively improves recall  
- Random Forest outperforms others  
- Amount alone does NOT predict fraud  
- ML reduces false negatives (most dangerous category)

---

## Dashboard / Model / Output

### **Streamlit App Features**
- Manual entry (30 features)
- CSV bulk prediction
- Fraud probability output
- Pie charts, bar charts, distribution plots
- MySQL logging

### **Power BI**
- Fraud vs Non-fraud distribution  
- Transaction amount patterns  
- PCA-based patterns  
- Risk behavior  

![Credit card dashboard](Dashboard/Dashboard.img.png)

---

## How to Run This Project

### **1. Clone Repository**
```bash
git clone https://github.com/Himanshupatel17/credit-card-fraud-detection-ml.git
2. Create Database
bash
Copy code
python scripts/create_database.py
3. Load Example Data (optional)
bash
Copy code
python scripts/load_to_mysql.py
4. Run Notebooks
Copy code
notebooks/01_EDA.ipynb
notebooks/02_Model_Training.ipynb
notebooks/03_Model_Comparison.ipynb
5. Run Streamlit App
bash
Copy code
streamlit run app.py
6. Open Power BI Dashboard
bash
Copy code
dashboard/fraud_detection_dashboard.pbix
Results & Conclusion
Random Forest achieved 99.97% accuracy

Fraud recall improved significantly after SMOTE

Full ML pipeline built end-to-end

Supports near real-time fraud prediction

Practical for real-world fintech/banking systems

Future Work
Real-time API integration

Deep learning models (Autoencoders, LSTM)

Cloud deployment (AWS / Azure / GCP)

Alerts for high-risk transactions

Explainable AI (SHAP, LIME)

Author & Contact
Himanshu Patel
Machine Learning & Data Science Enthusiast

📧 Email: hhimanshu714@gmail.com
🔗 GitHub: Himanshupatel17
🔗 LinkedIn: linkedin.com/in/himanshupatel1715