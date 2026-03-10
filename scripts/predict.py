import os
import sys
import pandas as pd
import joblib

# -----------------------
# Proje kökü
# -----------------------
project_root = r"C:\Users\esram\OneDrive\Desktop\esrabecode\immo-eliza-ml"
sys.path.append(project_root)

from scripts.random_forest_model import run_pipeline_pipeline5, load_and_clean_data

# -----------------------
# Model ve kolonların yolu
# -----------------------
model_path = os.path.join(project_root, "models", "random_forest.pkl")
train_cols_path = os.path.join(project_root, "models", "rf_train_columns.pkl")

if not os.path.exists(model_path) or not os.path.exists(train_cols_path):
    raise FileNotFoundError(
        f"Model veya kolon dosyası bulunamadı. Önce random_forest_model.py ile train etmelisin."
    )

# Model ve train kolonları yükle
rf_model = joblib.load(model_path)
train_columns = joblib.load(train_cols_path)

# -----------------------
# Yeni veri yükle
# -----------------------
data_file = os.path.join(project_root, "data", "immovlan_cleaned_file_final.csv")
df = load_and_clean_data(data_file)

# -----------------------
# Pipeline ile feature engineering ve encoding
# -----------------------
# Sadece test verisini encode ediyoruz, modeli tekrar train etmiyoruz
_, _, _, X_test_enc, _, _, _ = run_pipeline_pipeline5(
    df,
    target_col='Price',
    rf_params=None,
    feature_eng=True,
    rare_city=True,
    train_mode=False
)

# Test verisini train kolonlarına göre hizala
X_test_enc = X_test_enc.reindex(columns=train_columns, fill_value=0)

# -----------------------
# Tahmin yap
# -----------------------
y_pred = rf_model.predict(X_test_enc)

# Sonuçları göster
results = pd.DataFrame({
    "Predicted": y_pred
})
print(results.head())
