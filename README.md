# ChurnOps: End-to-End MLOps System for Telecom Customer Churn Prediction

## 1. Project Overview

**ChurnOps** is an end-to-end MLOps system for predicting telecom customer churn. The project demonstrates how a machine learning model can move beyond notebook experimentation into a reproducible and deployable ML system.

The system covers the full MLOps lifecycle:

```text
Business Problem
→ Public Dataset
→ Data Validation
→ Preprocessing
→ Train/Validation/Test Split
→ Baseline and Improved Model Training
→ MLflow Experiment Tracking
→ Model Evaluation
→ MLflow Model Registry
→ FastAPI Deployment
→ Docker Containerization
→ Monitoring and Drift Detection
→ Batch Prediction
→ Dashboard
→ CI/CD
→ Reproducibility
```

---

## 2. Business Problem

Telecom companies lose revenue when customers cancel their subscriptions. The business goal is to identify customers who are likely to churn so that the retention team can prioritize them for intervention.

Instead of sending retention offers to every customer, the company can focus on customers with high churn probability. This supports better campaign targeting, reduced wasted offers, and improved customer retention efficiency.

**Business decision supported:**

> Which customers should be prioritized for retention action?

---

## 3. Dataset

The project uses the public **Telco Customer Churn** dataset.

The raw dataset should be saved as:

```text
data/raw/telco_churn.csv
```

Expected columns include:

```text
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
Churn
```

The dataset is not committed to the repository. To reproduce the project, download the public Telco Customer Churn dataset and place it in the `data/raw/` folder using the filename `telco_churn.csv`.

---

## 4. Target, Features, and Metrics

### Target Variable

| Item           | Value                 |
| -------------- | --------------------- |
| Target         | `Churn`               |
| Type           | Binary classification |
| Positive class | `Yes`                 |
| Negative class | `No`                  |

### Feature Groups

| Feature Group       | Example Features                                                                     |
| ------------------- | ------------------------------------------------------------------------------------ |
| Customer profile    | `gender`, `SeniorCitizen`, `Partner`, `Dependents`                                   |
| Account information | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`                            |
| Services            | `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Billing             | `MonthlyCharges`, `TotalCharges`                                                     |

The `customerID` column is removed because it is an identifier and should not be used as a predictive feature.

### Evaluation Metrics

| Metric           | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| ROC-AUC          | Measures model ranking quality                        |
| Recall           | Measures how many actual churners were caught         |
| Precision        | Measures how many predicted churners actually churned |
| F1-score         | Balances precision and recall                         |
| Confusion matrix | Shows false positives and false negatives             |

The primary model selection metric is **ROC-AUC**.
The key business metric is **recall**, because missed churners may represent lost revenue.

---

## 5. System Architecture

```text
Raw Data
   ↓
Data Validation
   ↓
Preprocessing and Feature Engineering
   ↓
Train / Validation / Test Split
   ↓
Model Training
   ├── Logistic Regression Baseline
   └── Random Forest Improved Model
   ↓
MLflow Experiment Tracking
   ↓
Model Evaluation
   ↓
MLflow Model Registry
   ↓
FastAPI Model Serving
   ↓
Dockerized API
   ↓
Monitoring and Drift Detection
   ↓
Batch Prediction and Dashboard
```

---

## 6. Repository Structure

```text
churnops-mlops/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── batch/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingest_data.py
│   │   └── validate_data.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── register_model.py
│   │   ├── feature_importance.py
│   │   └── threshold_analysis.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── monitoring/
│   │   └── generate_drift_report.py
│   │
│   ├── batch/
│   │   └── batch_predict.py
│   │
│   └── dashboard/
│       └── app.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data_validation.py
│   └── test_model_prediction.py
│
├── configs/
│   └── config.yaml
│
├── reports/
├── models/
├── docker/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── README.md
└── .gitignore
```

---

## 7. Tools Used

| Tool           | Purpose                                |
| -------------- | -------------------------------------- |
| Python         | Main programming language              |
| Pandas         | Data loading and validation            |
| Scikit-learn   | Preprocessing, modeling, and metrics   |
| MLflow         | Experiment tracking and model registry |
| FastAPI        | Model serving API                      |
| Docker         | Containerized deployment               |
| Streamlit      | Business-facing dashboard              |
| Pytest         | Automated local tests                  |
| GitHub Actions | CI/CD checks                           |
| Matplotlib     | Plots and model evidence               |
| Joblib         | Model serialization                    |

---

## 8. Setup Instructions

### 8.1 Clone the Repository

```bash
git clone https://github.com/jaynadzlibardo/churnops-mlops.git
cd churnops-mlops
```

### 8.2 Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 8.3 Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 8.4 Add Dataset

Download the public Telco Customer Churn dataset and save it as:

```text
data/raw/telco_churn.csv
```

---

## 9. How to Run the Project

### 9.1 Data Validation

```bash
python src/data/validate_data.py
```

Output:

```text
reports/data_validation_report.json
```

This checks required columns, row count, missing values, duplicate rows, target distribution, and `TotalCharges` quality.

---

### 9.2 Preprocessing and Feature Engineering

```bash
python src/features/build_features.py
```

Outputs:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/preprocessor.pkl
reports/feature_metadata.json
```

The data is split using stratification. The preprocessor is fitted only on the training set to prevent data leakage.

---

### 9.3 Model Training with MLflow

```bash
python src/models/train_model.py
```

Models trained:

| Model               | Role           |
| ------------------- | -------------- |
| Logistic Regression | Baseline       |
| Random Forest       | Improved model |

Outputs:

```text
models/churn_model.pkl
reports/metrics_report.json
reports/confusion_matrix.png
mlruns/
```

---

### 9.4 Launch MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

MLflow shows model runs, parameters, metrics, artifacts, and registered model versions.

---

### 9.5 Final Test Evaluation

```bash
python src/models/evaluate_model.py
```

Outputs:

```text
reports/test_metrics_report.json
reports/test_confusion_matrix.png
```

Final test results from the current run:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.7474 |
| ROC-AUC   | 0.8446 |
| Precision | 0.5165 |
| Recall    | 0.7794 |
| F1-score  | 0.6213 |

Confusion matrix values:

| Type            | Count |
| --------------- | ----: |
| True Negatives  |   571 |
| False Positives |   205 |
| False Negatives |    62 |
| True Positives  |   219 |

Business interpretation:

* The model caught 219 churners.
* The model missed 62 churners.
* False negatives are costly because they represent customers who may leave without intervention.
* False positives may waste retention offers but are less severe if the retention campaign cost is low.

---

### 9.6 Register Best Model in MLflow

```bash
python src/models/register_model.py
```

Registered model:

```text
churnops_model
```

Alias:

```text
champion
```

Output:

```text
reports/model_registry_report.json
```

---

### 9.7 Run FastAPI Locally

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
```

Prediction request:

```powershell
$body = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = 5
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "Fiber optic"
    OnlineSecurity = "No"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "Yes"
    StreamingMovies = "Yes"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 95.70
    TotalCharges = 478.50
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType "application/json"
```

Example response:

```text
prediction        : 1
prediction_label  : Churn
churn_probability : 0.8845
risk_level        : High
business_action   : Prioritize for retention campaign or proactive customer support.
```

---

### 9.8 Docker Deployment

Build Docker image:

```bash
docker build -t churnops-api:latest .
```

Run Docker container:

```bash
docker run --name churnops-api-container -p 8000:8000 churnops-api:latest
```

Test API:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
```

Stop and remove container:

```bash
docker stop churnops-api-container
docker rm churnops-api-container
```

---

### 9.9 Monitoring and Drift Detection

```bash
python src/monitoring/generate_drift_report.py
```

Outputs:

```text
reports/monitoring_report.html
reports/monitoring_summary.json
```

Current monitoring result:

| Monitoring Item           | Result |
| ------------------------- | ------ |
| Features monitored        | 19     |
| Drifted features          | 0      |
| Drifted feature share     | 0.00%  |
| Data drift detected       | False  |
| Prediction drift detected | False  |
| Retraining recommended    | False  |
| Alert level               | OK     |

Open monitoring report:

```powershell
start reports/monitoring_report.html
```

Business interpretation:

> If drift is detected, the team should review data quality, investigate customer behavior changes, and consider retraining the model.

---

### 9.10 Batch Prediction

```bash
python src/batch/batch_predict.py
```

Input:

```text
data/batch/sample_customers.csv
```

Output:

```text
reports/batch_predictions.csv
```

Current batch result:

| Risk Level | Count |
| ---------- | ----: |
| High       |     2 |
| Medium     |     0 |
| Low        |     2 |

---

### 9.11 Feature Importance

```bash
python src/models/feature_importance.py
```

Outputs:

```text
reports/feature_importance.csv
reports/feature_importance.png
reports/feature_importance_report.json
```

Top churn drivers from the current model:

| Rank | Feature                     | Importance |
| ---: | --------------------------- | ---------: |
|    1 | tenure                      |     1.2068 |
|    2 | Contract_Two year           |     0.7461 |
|    3 | Contract_Month-to-month     |     0.6621 |
|    4 | TotalCharges                |     0.5874 |
|    5 | InternetService_Fiber optic |     0.5660 |
|    6 | InternetService_DSL         |     0.5068 |
|    7 | MonthlyCharges              |     0.5007 |

Note:

> Feature importance shows model influence, not guaranteed causality.

---

### 9.12 Threshold Analysis

```bash
python src/models/threshold_analysis.py
```

Outputs:

```text
reports/threshold_analysis.csv
reports/threshold_analysis.png
reports/threshold_analysis_report.json
```

Threshold analysis summary:

| Threshold | Precision | Recall |     F1 | Customers Flagged | Missed Churners |
| --------: | --------: | -----: | -----: | ----------------: | --------------: |
|      0.30 |    0.4319 | 0.9253 | 0.5889 |               602 |              21 |
|      0.40 |    0.4646 | 0.8399 | 0.5982 |               508 |              45 |
|      0.50 |    0.5165 | 0.7794 | 0.6213 |               424 |              62 |
|      0.60 |    0.5714 | 0.6975 | 0.6282 |               343 |              85 |
|      0.70 |    0.6274 | 0.5872 | 0.6066 |               263 |             116 |

Recommended threshold by F1:

```text
0.60
```

Business interpretation:

* Lower threshold catches more churners but increases campaign cost.
* Higher threshold reduces false positives but misses more churners.
* Threshold choice depends on retention budget and business risk tolerance.

---

### 9.13 Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open:

```text
http://localhost:8501
```

Dashboard sections:

1. Manual churn prediction
2. Final test metrics
3. Feature importance
4. Threshold analysis
5. Monitoring summary

---

### 9.14 Run Tests

```bash
pytest tests/
```

Current local test result:

```text
8 passed
```

The tests validate:

* Data files and processed splits
* Target column
* Model prediction behavior
* FastAPI `/health`
* FastAPI `/predict`

---

## 10. CI/CD

This project includes GitHub Actions:

```text
.github/workflows/ci.yml
```

The workflow checks:

1. Python setup
2. Dependency installation
3. Core imports
4. Repository structure

GitHub Actions status:

```text
Passed
```

The full model/API functional tests are run locally because data and model artifacts are intentionally not committed to Git.

---

## 11. Demo Walkthrough

Recommended demo order:

1. Show GitHub repository.
2. Show repository structure.
3. Show public dataset source.
4. Run data validation.
5. Show validation report.
6. Run preprocessing.
7. Show train/validation/test split.
8. Run model training.
9. Open MLflow dashboard.
10. Compare Logistic Regression and Random Forest.
11. Show selected best model.
12. Run final test evaluation.
13. Show confusion matrix and metrics.
14. Register model in MLflow.
15. Run FastAPI.
16. Open Swagger UI.
17. Test `/health`.
18. Test `/predict`.
19. Build Docker image.
20. Run Docker container.
21. Test API from Docker.
22. Run monitoring script.
23. Open monitoring report.
24. Run batch prediction.
25. Show feature importance.
26. Show threshold analysis.
27. Open Streamlit dashboard.
28. Show GitHub Actions passing.
29. End with business value, limitations, and next steps.

---

## 12. Demo Evidence Checklist

| Evidence                              | Status       |
| ------------------------------------- | ------------ |
| Public dataset source                 | Must-have    |
| GitHub repository                     | Must-have    |
| GitHub Actions passed                 | Must-have    |
| Local pytest passed                   | Must-have    |
| Data validation logs                  | Must-have    |
| Data validation JSON report           | Must-have    |
| Preprocessing logs                    | Must-have    |
| Train/validation/test split files     | Must-have    |
| MLflow experiment dashboard           | Must-have    |
| Baseline vs improved model comparison | Must-have    |
| MLflow model registry                 | Must-have    |
| Test metrics report                   | Must-have    |
| Confusion matrix                      | Must-have    |
| FastAPI Swagger UI                    | Must-have    |
| `/health` response                    | Must-have    |
| `/predict` response                   | Must-have    |
| Docker build logs                     | Must-have    |
| Docker container running              | Must-have    |
| Docker API response                   | Must-have    |
| Monitoring report                     | Must-have    |
| Batch prediction output               | Nice-to-have |
| Feature importance chart              | Nice-to-have |
| Threshold analysis chart              | Nice-to-have |
| Streamlit dashboard                   | Nice-to-have |

---

## 13. Key Results

Final selected model:

```text
Logistic Regression baseline
```

Reason:

> It achieved the best validation ROC-AUC.

Final test ROC-AUC:

```text
0.8446
```

Final test recall:

```text
0.7794
```

Business meaning:

> The model successfully catches about 78% of actual churners on the test set, helping the retention team reduce missed churn-risk customers.

---

## 14. Limitations

1. The dataset is static and does not represent a real-time production stream.
2. The model predicts churn risk but does not prove causality.
3. Actual business value depends on the effectiveness and cost of the retention campaign.
4. Monitoring is simulated using validation and test splits.
5. Real production monitoring would require new incoming customer data and actual churn outcomes.
6. The model artifact and dataset are not committed to Git, so users must rerun the pipeline locally.

---

## 15. Future Work

Possible upgrades:

1. Deploy FastAPI to a cloud service.
2. Add DVC for data versioning.
3. Add Prometheus and Grafana for API monitoring.
4. Add automated retraining when drift persists.
5. Add batch scoring from cloud storage.
6. Add fairness or bias analysis.
7. Add customer lifetime value to prioritize high-value churners.
8. Add campaign ROI calculation.
9. Add model explainability using SHAP.
10. Add scheduled monitoring jobs.

---

## 16. Business Value

This system helps convert churn prediction into an operational decision system.

Instead of only reporting model accuracy, ChurnOps provides:

* Customer-level churn probabilities
* Business-friendly risk levels
* Retention action recommendations
* Batch scoring for campaign planning
* API serving for application integration
* Monitoring to detect drift
* Reproducibility through scripts, Docker, tests, and CI/CD

The main business impact is better retention targeting, reduced wasted campaign effort, and lower risk of losing high-risk customers without intervention.
