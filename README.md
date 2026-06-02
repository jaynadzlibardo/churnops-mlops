# ChurnOps: End-to-End MLOps System for Telecom Customer Churn Prediction

## Business Problem

Telecom companies lose revenue when customers cancel their subscriptions. This project predicts customer churn risk so the retention team can prioritize high-risk customers for intervention.

## Project Goal

Build an end-to-end MLOps system covering:

1. Data validation
2. Preprocessing
3. Train/validation/test split
4. Baseline and improved model training
5. MLflow experiment tracking
6. Model evaluation
7. FastAPI deployment
8. Docker containerization
9. Monitoring with Evidently AI
10. Reproducibility documentation

## Target Variable

`Churn`

- Positive class: `Yes`
- Negative class: `No`

## Main Metrics

- ROC-AUC
- Recall
- Precision
- F1-score
- Confusion matrix

## Business Interpretation

High recall means fewer missed churners.  
High precision means fewer wasted retention offers.

## Current Status

Phase 1: Project setup.