# ChurnOps: End-to-End MLOps System for Telecom Customer Churn Prediction

## 1. Project Overview

**ChurnOps** is an end-to-end MLOps system for predicting telecom customer churn. The project demonstrates how a machine learning model can move beyond notebook experimentation into a reproducible, testable, deployable, and monitorable ML system.

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
→ Streamlit Dashboard
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

## 3. Project Goal

Build an end-to-end MLOps system that can:

1. Validate raw customer churn data.
2. Preprocess data without leakage.
3. Train and compare machine learning models.
4. Track experiments and metrics using MLflow.
5. Register the best model as a champion model.
6. Serve predictions through FastAPI.
7. Containerize the API with Docker.
8. Monitor data drift and prediction drift.
9. Run batch prediction for customer retention prioritization.
10. Present model results through a Streamlit dashboard.
11. Validate code quality through Pytest and GitHub Actions.

---

## 4. Dataset

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

The dataset is not committed to the repository. To reproduce the project, download the public Telco Customer Churn dataset and place it in the `data/raw/` folder using the filename:

```text
telco_churn.csv
```

---

## 5. Target, Features, and Metrics

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

## 6. System Architecture

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
Batch Prediction
   ↓
Streamlit MLOps Dashboard
   ↓
CI/CD and Reproducibility
```

---

## 7. Repository Structure

```text
churnops-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   └── config.yaml
│
├── dashboard/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── batch/
│
├── docker/
│
├── mlruns/
│
├── models/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── batch/
│   │   └── batch_predict.py
│   │
│   ├── data/
│   │   ├── ingest_data.py
│   │   └── validate_data.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── evaluate_model.py
│   │   ├── feature_importance.py
│   │   ├── register_model.py
│   │   ├── threshold_analysis.py
│   │   └── train_model.py
│   │
│   └── monitoring/
│       └── generate_drift_report.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data_validation.py
│   └── test_model_prediction.py
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 8. Tools Used

| Tool           | Purpose                                      |
| -------------- | -------------------------------------------- |
| Python         | Main programming language                    |
| Pandas         | Data loading, transformation, and validation |
| Scikit-learn   | Preprocessing, modeling, and metrics         |
| MLflow         | Experiment tracking and model registry       |
| FastAPI        | Model serving API                            |
| Docker         | Containerized deployment                     |
| Streamlit      | Business-facing MLOps dashboard              |
| Pytest         | Automated local tests                        |
| GitHub Actions | CI/CD checks                                 |
| Matplotlib     | Dashboard and model evidence charts          |
| Joblib         | Model serialization                          |

---

## 9. Setup Instructions

### 9.1 Clone the Repository

```bash
git clone https://github.com/jaynadzlibardo/churnops-mlops.git
cd churnops-mlops
```

### 9.2 Create a Virtual Environment

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

### 9.3 Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.4 Add Dataset

Download the public Telco Customer Churn dataset and save it as:

```text
data/raw/telco_churn.csv
```

---

## 10. How to Run the Project

### 10.1 Data Validation

```bash
python src/data/validate_data.py
```

Output:

```text
reports/data_validation_report.json
```

This checks:

* Required columns
* Row count
* Missing values
* Duplicate rows
* Target distribution
* `TotalCharges` quality

---

### 10.2 Preprocessing and Feature Engineering

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

### 10.3 Model Training with MLflow

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

### 10.4 Launch MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

MLflow shows:

* Model runs
* Parameters
* Metrics
* Artifacts
* Registered model versions

---

### 10.5 Final Test Evaluation

```bash
python src/models/evaluate_model.py
```

Outputs:

```text
reports/test_metrics_report.json
reports/test_confusion_matrix.png
```

Final selected model:

```text
Logistic Regression baseline
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

* The model caught **219 churners**.
* The model missed **62 churners**.
* False negatives are costly because they represent customers who may leave without intervention.
* False positives may waste retention offers but are less severe if the retention campaign cost is lower than the cost of losing a customer.

---

### 10.6 Register Best Model in MLflow

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

### 10.7 Run FastAPI Locally

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

### 10.8 Docker Deployment

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

### 10.9 Monitoring and Drift Detection

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

### 10.10 Batch Prediction

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

Business interpretation:

> The retention team should prioritize high-risk customers for immediate outreach.

---

### 10.11 Feature Importance

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

Business interpretation:

| Driver                  | Possible Business Action                                             |
| ----------------------- | -------------------------------------------------------------------- |
| Low tenure              | Improve onboarding and early-life customer support                   |
| Month-to-month contract | Offer loyalty discounts or contract upgrade incentives               |
| High monthly charges    | Review pricing, bundles, and value communication                     |
| Fiber optic service     | Investigate service quality, speed, pricing, and customer complaints |
| High total charges      | Prioritize high-value customers for proactive retention              |

Note:

> Feature importance shows model influence, not guaranteed causality.

---

### 10.12 Threshold Analysis

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

Recommended initial policy:

> Use a `0.60` threshold for the first retention campaign, then adjust based on retention budget, campaign conversion, and cost per offer.

---

### 10.13 Streamlit MLOps Dashboard

Run the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit dashboard acts as a business-facing MLOps control room. It converts model outputs, monitoring results, threshold analysis, and batch scoring into decision-ready views for retention prioritization.

Dashboard pages:

| Page                | Purpose                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| Executive Overview  | Shows model performance, monitoring health, batch scoring summary, and project health |
| Model Performance   | Shows final test metrics and business interpretation                                  |
| Confusion Matrix    | Shows true positives, false positives, false negatives, and business error analysis   |
| Threshold Analysis  | Shows decision threshold trade-offs and recommended retention policy                  |
| Feature Importance  | Shows top churn drivers and business actions                                          |
| Monitoring & Drift  | Shows drift status, retraining decision, and monitoring policy                        |
| Batch Predictions   | Shows scored customers, risk bands, and retention action mapping                      |
| MLOps System Health | Shows production-readiness checklist and architecture flow                            |

### Dynamic Dashboard Inputs

The dashboard loads values dynamically from generated report files in the `reports/` folder.

| Dashboard Input             | Source File                          |
| --------------------------- | ------------------------------------ |
| Final model metrics         | `reports/test_metrics_report.json`   |
| Confusion matrix            | `reports/test_metrics_report.json`   |
| Monitoring and drift status | `reports/monitoring_summary.json`    |
| Batch prediction results    | `reports/batch_predictions.csv`      |
| Feature importance          | `reports/feature_importance.csv`     |
| Threshold analysis          | `reports/threshold_analysis.csv`     |
| Model registry information  | `reports/model_registry_report.json` |

Before running the dashboard, generate the latest reports by running the evaluation, monitoring, batch prediction, feature importance, and threshold analysis scripts.

Recommended report generation flow:

```bash
python src/models/evaluate_model.py
python src/monitoring/generate_drift_report.py
python src/batch/batch_predict.py
python src/models/feature_importance.py
python src/models/threshold_analysis.py
```

The dashboard helps answer:

* Is the champion model performing well?
* How many churners were captured or missed?
* What threshold should the business use?
* What features are driving churn?
* Is there data drift or prediction drift?
* Should the model be retrained?
* Which customers should be prioritized for retention?
* Is the MLOps system production-ready as a portfolio demo?

Current dashboard status:

| Dashboard Page      | Status   |
| ------------------- | -------- |
| Executive Overview  | Complete |
| Model Performance   | Complete |
| Confusion Matrix    | Complete |
| Threshold Analysis  | Complete |
| Feature Importance  | Complete |
| Monitoring & Drift  | Complete |
| Batch Predictions   | Complete |
| MLOps System Health | Complete |

Business value:

> The dashboard makes ChurnOps easier to demo because it connects model performance, business trade-offs, monitoring health, and customer-level retention actions in one interface.

---

### 10.14 Run Tests

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

## 11. CI/CD

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

## 12. Demo Walkthrough

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

## 13. Dashboard Demo Flow

Recommended dashboard demo order:

1. **Executive Overview**

   * Show ROC-AUC, recall, monitoring status, and project health.

2. **Model Performance**

   * Explain why recall and ROC-AUC matter for churn prioritization.

3. **Confusion Matrix**

   * Highlight that the model caught 219 churners and missed 62.

4. **Threshold Analysis**

   * Explain the business trade-off between campaign cost and missed churners.

5. **Feature Importance**

   * Translate churn drivers into retention actions.

6. **Monitoring & Drift**

   * Show that 19 features were monitored, 0 drifted, and retraining is not needed.

7. **Batch Predictions**

   * Show the retention queue: 2 high-risk customers, 0 medium-risk, 2 low-risk.

8. **MLOps System Health**

   * Close by showing the system is not just a notebook. It includes testing, CI, registry, API, Docker, monitoring, batch scoring, and dashboard reporting.

---

## 14. Demo Evidence Checklist

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
| Dashboard screenshots                 | Nice-to-have |

---

## 15. Key Results

Final selected model:

```text
Logistic Regression baseline
```

Reason:

> It achieved the best validation ROC-AUC and provides interpretability for business users.

Final test ROC-AUC:

```text
0.8446
```

Final test recall:

```text
0.7794
```

Final test F1-score:

```text
0.6213
```

Business meaning:

> The model successfully catches about 78% of actual churners on the test set, helping the retention team reduce missed churn-risk customers.

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
* Threshold tuning for retention budget control
* Dashboard reporting for business stakeholders
* Reproducibility through scripts, Docker, tests, and CI/CD

The main business impact is better retention targeting, reduced wasted campaign effort, and lower risk of losing high-risk customers without intervention.

---

## 17. Limitations

1. The dataset is static and does not represent a real-time production stream.
2. The model predicts churn risk but does not prove causality.
3. Actual business value depends on the effectiveness and cost of the retention campaign.
4. Monitoring is simulated using validation and test splits.
5. Real production monitoring would require new incoming customer data and actual churn outcomes.
6. The model artifact and dataset are not committed to Git, so users must rerun the pipeline locally.
7. The dashboard depends on generated files in the `reports/` folder. Users must run the pipeline scripts first before the dashboard can display the latest metrics, monitoring results, batch predictions, feature importance, and threshold analysis.

---

## 18. Future Work

Possible upgrades:

1. Add automated dashboard validation tests to confirm that all report files load correctly.
2. Add dashboard screenshots to the README.
3. Deploy FastAPI to a cloud service.
4. Deploy Streamlit dashboard to a hosted environment.
5. Add DVC for data versioning.
6. Add Prometheus and Grafana for API monitoring.
7. Add automated retraining when drift persists.
8. Add batch scoring from cloud storage.
9. Add fairness or bias analysis.
10. Add customer lifetime value to prioritize high-value churners.
11. Add campaign ROI calculation.
12. Add model explainability using SHAP.
13. Add scheduled monitoring jobs.
14. Add model comparison dashboard for champion vs challenger models.

---

## 19. Portfolio Summary

**ChurnOps** demonstrates a production-style MLOps workflow for telecom churn prediction.

The project includes:

* Data validation
* Reproducible preprocessing
* Train/validation/test split
* Baseline and improved model training
* MLflow experiment tracking
* MLflow model registry
* Champion model selection
* FastAPI model serving
* Dockerized deployment
* Monitoring and drift detection
* Batch prediction
* Feature importance
* Threshold analysis
* Streamlit dashboard
* Pytest validation
* GitHub Actions CI/CD

This project shows how machine learning outputs can be translated into business decisions for customer retention.
