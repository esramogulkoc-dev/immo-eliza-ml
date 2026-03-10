# =======================
# random_forest_model.py
# =======================

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# -----------------------
# Project root ayarı
# -----------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.clean_data import clean_immo_data

# -----------------------
# Veri yükleme ve temizleme
# -----------------------
def load_and_clean_data(file_path):
    df = clean_immo_data(file_path, save_file=False, verbose=True)
    df = df.dropna(subset=['Price'])
    return df

# -----------------------
# Pipeline fonksiyonu
# -----------------------
def run_pipeline_pipeline5(
    df, 
    target_col='Price', 
    rf_params=None,        # Random Forest parameters
    feature_eng=True,      # Apply feature engineering
    rare_city=True,        # Handle rare cities
    train_mode=True        # Train mi? Yoksa sadece predict mi?
):
    df = df[df[target_col] > 1]

    # Drop unwanted columns
    drop_cols = ['Property ID', 'url','Availability','Attic','Kitchen equipment','Kitchen type',
                 'Furnished','Price_per_sqm_land','price_per_sqm','Number of facades',
                 'Number of bathrooms','Number of showers','Number of toilets',
                 'Surface garden','Type of glazing']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Split data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train_df = pd.DataFrame(X_train, columns=X.columns, index=y_train.index)
    X_val_df   = pd.DataFrame(X_val, columns=X.columns, index=y_val.index)
    X_test_df  = pd.DataFrame(X_test, columns=X.columns, index=y_test.index)

    # -----------------------
    # Missing value handling
    # -----------------------
    zero_cols = ['Garage', 'Number of garages', 'Swimming pool', 'Terrace', 'Surface terrace','Elevator','Garden']
    categorical_cols = ['Type of heating', 'State of the property']

    for col in zero_cols:
        for df_ in [X_train_df, X_val_df, X_test_df]:
            df_[col].fillna(0, inplace=True)
    for col in categorical_cols:
        for df_ in [X_train_df, X_val_df, X_test_df]:
            df_[col].fillna('unknown', inplace=True)
    for df_ in [X_train_df, X_val_df, X_test_df]:
        df_['main_type'].fillna('unknown', inplace=True)

    # Total land surface
    X_train_df.loc[X_train_df['main_type']=='apartment','Total land surface'].fillna(0, inplace=True)
    X_val_df.loc[X_val_df['main_type']=='apartment','Total land surface'].fillna(0, inplace=True)
    X_test_df.loc[X_test_df['main_type']=='apartment','Total land surface'].fillna(0, inplace=True)

    house_median = X_train_df.loc[X_train_df['main_type']=='house','Total land surface'].median()
    land_median  = X_train_df.loc[X_train_df['main_type']=='land','Total land surface'].median()
    for df_ in [X_train_df, X_val_df, X_test_df]:
        df_.loc[(df_['main_type']=='house') & (df_['Total land surface'].isna()), 'Total land surface'] = house_median
        df_.loc[(df_['main_type']=='land') & (df_['Total land surface'].isna()), 'Total land surface'] = land_median

    # Numeric columns median fill
    num_cols_for_median = ['Number of bedrooms', 'Livable surface', 'Total land surface']
    for col in num_cols_for_median:
        median_dict = X_train_df.groupby('main_type')[col].median()
        for mtype, med_val in median_dict.items():
            for df_ in [X_train_df, X_val_df, X_test_df]:
                df_.loc[(df_['main_type']==mtype) & (df_[col].isna()), col] = med_val
        median_val = X_train_df[col].median()
        for df_ in [X_train_df, X_val_df, X_test_df]:
            df_[col].fillna(median_val, inplace=True)

    # Outlier removal
    outlier_num_cols = ['Number of bedrooms', 'Livable surface', 'Garage', 'Number of garages', 
                        'Terrace', 'Surface terrace', 'Total land surface', 'Swimming pool']

    if train_mode:
        df_normal_train_list = []
        y_train = y_train.reindex(X_train_df.index)
        for mtype, group in X_train_df.groupby('main_type'):
            group_copy = group.copy()
            group_copy[target_col] = y_train.loc[group_copy.index]
            z_scores = np.abs(stats.zscore(group_copy[outlier_num_cols], nan_policy='omit'))
            outliers = (z_scores > 3).any(axis=1)
            df_normal_train_list.append(group_copy[~outliers])
        df_normal_train = pd.concat(df_normal_train_list)
        X_tr = df_normal_train.drop(target_col, axis=1)
        y_tr = df_normal_train[target_col]
        X_val_copy = X_val_df.copy()
        X_test_copy = X_test_df.copy()
    else:
        # predict modu: train sadece X_test
        X_tr = y_tr = X_val_copy = None
        X_test_copy = X_test_df.copy()

    # Rare cities
    freq = None
    if rare_city and train_mode:
        freq = X_tr['city'].value_counts(normalize=True)
        rare_cities = freq[freq < 0.01].index
        for df_ in [X_tr, X_val_copy, X_test_copy]:
            df_['city'] = df_['city'].apply(lambda x: 'other' if x in rare_cities else x)
    elif rare_city and not train_mode:
        freq = X_test_copy['city'].value_counts(normalize=True)
        rare_cities = freq[freq < 0.01].index
        for df_ in [X_test_copy]:
            df_['city'] = df_['city'].apply(lambda x: 'other' if x in rare_cities else x)

    # Feature engineering
    feat_eng_cols = ['has_swimming_pool', 'has_garden', 'has_terrace', 'surface_ratio', 'area_per_bedroom']
    if feature_eng:
        target_dfs = [X_tr, X_val_copy, X_test_copy] if train_mode else [X_test_copy]
        for df_ in target_dfs:
            df_['has_swimming_pool'] = df_['Swimming pool'].apply(lambda x: 1 if x > 0 else 0)
            df_['has_garden'] = df_['Garden'].apply(lambda x: 1 if x > 0 else 0)
            df_['has_terrace'] = df_['Terrace'].apply(lambda x: 1 if x > 0 else 0)
            df_['surface_ratio'] = df_['Livable surface'] / df_['Total land surface'].replace(0,1)
            df_['area_per_bedroom'] = df_['Livable surface'] / df_['Number of bedrooms'].replace(0,1)

    # Categorical encoding
    cat_cols = ['State of the property', 'Type of heating','type', 'city', 'Region', 'province', 'main_type']
    cat_cols_existing = [c for c in cat_cols if c in X_test_copy.columns]

    if train_mode:
        X_tr_encoded  = pd.get_dummies(X_tr, columns=cat_cols_existing, drop_first=True)
        X_val_encoded = pd.get_dummies(X_val_copy, columns=cat_cols_existing, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_copy, columns=cat_cols_existing, drop_first=True)

    if feature_eng:
        for col in feat_eng_cols:
            if train_mode:
                X_tr_encoded[col]  = X_tr[col]
                X_val_encoded[col] = X_val_copy[col]
            X_test_encoded[col] = X_test_copy[col]

    if train_mode:
        # Align columns
        train_cols = X_tr_encoded.columns
        X_val_encoded  = X_val_encoded.reindex(columns=train_cols, fill_value=0)
        X_test_encoded = X_test_encoded.reindex(columns=train_cols, fill_value=0)

        # Train Random Forest
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **(rf_params or {}))
        model = rf_model.fit(X_tr_encoded, y_tr)

        # Evaluation
        y_train_pred = model.predict(X_tr_encoded)
        y_val_pred   = model.predict(X_val_encoded)
        y_test_pred  = model.predict(X_test_encoded)

        def evaluate(y_true, y_pred, label="Set"):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"{label} → RMSE: {rmse:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

        evaluate(y_tr, y_train_pred, "Train")
        evaluate(y_val, y_val_pred, "Validation")
        evaluate(y_test, y_test_pred, "Test")

        return model, X_tr_encoded, X_val_encoded, X_test_encoded, y_tr, y_val, y_test
    else:
        return None, None, None, X_test_encoded, None, None, None
