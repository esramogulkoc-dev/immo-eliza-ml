# main.py

"""
Main script to run the Immo Eliza ML pipeline.
This will:
1. Run all scripts: clean_data, linear_model, random_forest_model, xgboost
2. Evaluate models
"""

# -----------------------------
# 1️⃣ Import scripts
# -----------------------------
import clean_data  
import linear_model  
import random_forest_model 
import xgboost  

# -----------------------------
# 2️⃣ Run the pipelines
# -----------------------------
print("Running Linear Regression pipeline...")
linear_model.run_linear_model() 

print("Running Random Forest pipeline...")
random_forest_model.run_rf_model()  

print("Running XGBoost pipeline...")
xgboost.run_xgb_pipeline()  

print("All pipelines executed successfully!")
