# Immo Eliza ML Project

## Project Overview
This repository contains a machine learning pipeline for predicting real estate prices in Belgium. The project focuses on data preprocessing, model training, and evaluation with multiple regression models.

---

## Repository: `immo-eliza-ml`
- **Type:** Consolidation  
- **Duration:** 4 days  
- **Deadline:** 27/11/2024 5:00 PM  
- **Show and Tell:** 01/12/2024 9:30 - 10:30 AM  
- **Team:** Solo

---

## Learning Objectives
- Preprocessed data for machine learning (missing values, encoding, scaling).  
- Applied Linear Regression as a baseline.  
- Explored advanced models (XGBoost, Random Forest).  
- Evaluated model performance with RMSE, MAE, R².  
- Applied hyperparameter tuning and cross-validation.

---

## Dataset Preparation
The real estate dataset was cleaned and preprocessed:

- **Drop low-value entries:** `Price <= 1`.  
- **Handle missing values:** numeric columns filled with median/zero, categorical with `'unknown'`.  
- **Outlier removal:** z-score based, applied per property type before train-test split.  
- **Feature engineering:**  
  - `has_swimming_pool`, `has_garden`, `has_terrace`  
  - `surface_ratio`, `area_per_bedroom`  
- **Categorical encoding:** one-hot, rare city categories grouped as `other`.  
- **Scaling:** StandardScaler for numeric features.  
- **Log-transform:** target variable `Price` for linear regression.

---

## Models

### 1. Linear Regression
- Baseline model with log-transformed target.  
- Evaluated on Train, Validation, and Test sets.  
- Metrics: RMSE, R².  

### 2. XGBoost Regressor
- Two pipelines tested:  
  1. **Simple XGBoost** with default tuned parameters.  
  2. **GridSearchCV XGBoost** for hyperparameter optimization.  
- Feature-engineered and one-hot encoded inputs.  
- Evaluated with RMSE, MAE, R².

### 3. Random Forest Regressor
- Seven pipelines tested with different preprocessing and feature sets.  
- **Pipeline 5** achieved the best performance and was selected as the final Random Forest model.  
- Captures non-linear relationships and handles both numeric and categorical features.  
- Evaluated using RMSE, MAE, and R².

---

## Usage
1. **Train models:** Use `random_forest_model.py` to preprocess data and train models (this will create `random_forest.pkl`).  
2. **Predict new data:** Use `predict.py` to load the saved model and predict property prices.  
3. **Evaluation:** Metrics printed after training; check for overfitting using Train vs Validation performance.
