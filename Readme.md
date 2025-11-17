# Credit Card Fraud Detection | Machine Learning Project

_Detecting fraudulent financial transactions using ML, SQL, Streamlit, and Power BI._

---
## 📌 Table of Contents
-<a href="#overview">Overview</a>
-<a href="#problem-statement">Problem Statement</a>
-<a href="#dataset">Dataset</a>
-<a href="#tools-and-technologies">Tools & Technologies</a>
-<a href="#project-structure">project structure</a>
-<a href="#methods">Methods</a>
-<a href="#key-insights">Key Insights</a>
-<a href="#dashboard-model-output">Dashboard / Model / Output</a>
-<a href="#how-to-run-this-project">How to Run This Project</a>
-<a href="#results--conclusion">Results & Conclusion</a>
-<a href="#future-work">Future Work</a>
-<a href="#author--contact">Author & Contact</a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>

This project builds an end-to-end ML system for identifying fraudulent credit card transactions.
The solution includes data preprocessing, SMOTE balancing, training multiple machine learning models, deploying the best model using Streamlit, storing prediction logs in MySQL, and creating an analytical dashboard in Power BI.
It replicates a real-world financial fraud detection workflow used in banks and fintech companies.

---
<h2><a class="anchor" id="problem-statement"></a> Problem Statement</h2>

-Credit card fraud causes massive financial losses.
-Traditional rule-based systems fail because:
-Fraud patterns change quickly
-Attackers adapt to rules
-Rules cannot detect non-obvious anomalies

---
Goal:
Build an ML-based system that accurately predicts whether a transaction is fraudulent using PCA-transformed features.

---
<h2><a class="anchor" id="dataset"></a> Dataset</h2>

-Source: Kaggle – Credit Card Fraud Detection
-Rows: 284,807
-Features:
-Time, Amount
-V1–V28 (PCA transformed)
-Class (0 = genuine, 1 = fraud)

Imbalance:
-Fraud = 492 cases (0.17%) → solved using SMOTE.

---
<h2><a class="anchor" id="tools-and-technologies"></a> Tools & Technologies</h2>

-Python (Pandas, NumPy, Scikit-learn, XGBoost, imblearn)
-Streamlit (Deployment UI)
-MySQL (Prediction logging)
-SQLAlchemy, PyMySQL
-Power BI (Dashboard)
-Git & GitHub

---
<h2><a class="anchor" id="project-structure"></a>project structure</h2>

```
credit-card-fraud-detection-ml/
│
├── app.py                 # Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Required dependencies
│
├── data/                  # Dataset storage
│   ├── raw/               # Raw dataset
│   ├── processed/         # Preprocessed train/test data
│
├── notebooks/             # Jupyter notebooks
│   ├── EDA, model training, comparison
│
├── models/                # Saved ML models (.pkl)
│
├── scripts/               # Automation scripts
│
├── reports/               # Final PDF + PPT + screenshots
│
└── docs/                  # Additional documentation
```

---
<h2 id="methods">🔬 Methods</h2>
-Data Preprocessing
-Cleaned dataset
-Standardization
-SMOTE oversampling for balance

--Model Training

Trained 3 models:

-Logistic Regression
-Random Forest
-XGBoost

Random Forest performed the best → deployed.
-Deployment
-Saved model (.pkl)
-Built Streamlit interface
-Manual + CSV prediction
-Stored predictions in MySQL

---
<h2><a class="anchor" id="key-insights"></a> Key Insights</h2>

-Fraud and genuine transactions show clear PCA separation
-SMOTE significantly improves fraud recall
-Random Forest yields highest overall performance
-ML reduces false negatives (most dangerous type)
-Amount alone doesn’t identify fraud

---
<h2><a class="anchor" id="dashboard-model-output"></a> Dashboard / Model / Output</h2>

**Streamlit App:**

-30-feature manual prediction
-CSV bulk prediction
-Pie charts, bar charts, distribution plots
-MySQL logging
-MySQL
-Stores all predictions with all 30 features.

**Power BI:**

Shows:

-Fraud vs Legit distribution
-Amount trends
-Risk patterns

![Credit card dashboard](Dashboard/Dashboard.img.png)

---
<h2><a class="anchor" id="How-to-run-this-project"></a> how to run this project</h2>

1️. Clone Repo:
 ```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection-ml.git
 ```
2. Create the MySQL Database:
 ```bash
python scripts/create_database.py
 ```
3. Load Example Data / Predictions into MySQL (optional)
 ```bash
python scripts/load_to_mysql.py
 ```

4. Open and Run Analytical Notebooks
 ```bash
-notebooks/01_EDA.ipynb
-notebooks/02_Model_Training.ipynb
-notebooks/03_Model_Comparison.ipynb
 ```

4. Run Streamlit App
 ```bash
streamlit run app.py
 ```

5. Open the Power BI Dashboard
 ```bash
dashboard/fraud_detection_dashboard.pbix
 ```

---
<h2><a class="anchor" id="Result-and-conclusion"></a> result and conclusion</h2>
**Results & Conclusion:**

-Random Forest achieved 99.97% accuracy
-Fraud recall significantly improved after SMOTE
-Complete working ML pipeline built end-to-end
-System supports real-time-like fraud prediction
-Accurate AND practical for real business use cases

**Future Work:**

-Real-time API-based fraud detection
-Deep learning models (Autoencoders, LSTM)
-Deploy to AWS/Azure/GCP
-Notification system for high-risk transactions

**Improve explainability (SHAP values, LIME):**

---
<h2><a class="anchor" id="author--contact"></a> author and contact</h2>

**Himanshu Patel:**
Machine Learning & Data Science Enthusiast

📧 Email: hhimanshu714@gmail.com
🔗 [GitHub] (https://github.com/Himanshupatel17/credit-card-fraud-detection-ml)
🔗 [LinkedIn] (www.linkedin.com/in/himanshupatel1715)