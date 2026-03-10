import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

def run_linear_pipeline(df, log_transform=True):
    """
    Linear Regression Pipeline:
    - Outlier removal
    - Missing value handling
    - Train/Val/Test split
    - Encoding categorical variables
    - Scaling numeric features
    - Optional log-transform of target
    """
    # -----------------------------
    # Temel filtreleme
    # -----------------------------
    df = df[df['Price'] > 1]

    drop_cols = [
        'Property ID', 'url', 'Availability', 'Attic', 'Kitchen equipment',
        'Kitchen type', 'Furnished', 'Price_per_sqm_land', 'price_per_sqm',
        'Number of facades', 'Number of bathrooms', 'Number of showers',
        'Number of toilets', 'Surface garden', 'Type of glazing'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # -----------------------------
    # Outlier removal (z-score)
    # -----------------------------
    outlier_num_cols = ['Number of bedrooms', 'Livable surface', 'Garage', 'Number of garages',
                        'Terrace', 'Surface terrace', 'Total land surface', 'Swimming pool']
    df_no_outliers_list = []
    for mtype, group in df.groupby('main_type'):
        group_copy = group.copy()
        z_scores = np.abs(stats.zscore(group_copy[outlier_num_cols], nan_policy='omit'))
        outliers = (z_scores > 3).any(axis=1)
        df_no_outliers_list.append(group_copy[~outliers])
    df = pd.concat(df_no_outliers_list)

    # -----------------------------
    # Missing value handling
    # -----------------------------
    zero_cols = ['Garage', 'Number of garages', 'Swimming pool', 'Terrace', 'Surface terrace', 'Elevator', 'Garden']
    for col in zero_cols:
        df[col].fillna(0, inplace=True)

    categorical_cols = ['Type of heating', 'State of the property']
    for col in categorical_cols:
        df[col].fillna('unknown', inplace=True)

    df['main_type'].fillna('unknown', inplace=True)

    # Special handling for Total land surface
    df.loc[df['main_type']=='apartment', 'Total land surface'].fillna(0, inplace=True)
    house_median = df.loc[df['main_type']=='house', 'Total land surface'].median()
    land_median  = df.loc[df['main_type']=='land', 'Total land surface'].median()
    df.loc[(df['main_type']=='house') & (df['Total land surface'].isna()), 'Total land surface'] = house_median
    df.loc[(df['main_type']=='land') & (df['Total land surface'].isna()), 'Total land surface'] = land_median

    num_cols_for_median = ['Number of bedrooms', 'Livable surface', 'Total land surface']
    median_dicts = df.groupby('main_type')[num_cols_for_median].median()
    for col in num_cols_for_median:
        for mtype, med_val in median_dicts[col].items():
            df.loc[(df['main_type']==mtype) & (df[col].isna()), col] = med_val
    for col in num_cols_for_median:
        df[col].fillna(df[col].median(), inplace=True)

    # -----------------------------
    # Train/Val/Test split
    # -----------------------------
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # -----------------------------
    # Categorical encoding
    # -----------------------------
    cat_cols = ['State of the property', 'Type of heating','type', 'city', 'Region', 'province', 'main_type']
    cat_cols_existing = [c for c in cat_cols if c in X_train.columns]

    X_tr_encoded = pd.get_dummies(X_train, columns=cat_cols_existing, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_cols_existing, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols_existing, drop_first=True)

    # Align columns
    train_cols = X_tr_encoded.columns
    X_val_encoded = X_val_encoded.reindex(columns=train_cols, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=train_cols, fill_value=0)

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    num_cols_scaled = ['Number of bedrooms', 'Livable surface', 'Number of garages',
                       'Surface terrace', 'Total land surface']
    X_tr_encoded[num_cols_scaled] = scaler.fit_transform(X_tr_encoded[num_cols_scaled])
    X_val_encoded[num_cols_scaled] = scaler.transform(X_val_encoded[num_cols_scaled])
    X_test_encoded[num_cols_scaled] = scaler.transform(X_test_encoded[num_cols_scaled])

    # -----------------------------
    # Log-transform target
    # -----------------------------
    if log_transform:
        y_tr_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        y_test_log = np.log1p(y_test)
    else:
        y_tr_log = y_train
        y_val_log = y_val
        y_test_log = y_test

    # -----------------------------
    # Train Linear Regression
    # -----------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_tr_encoded, y_tr_log)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_train_pred_log = lr_model.predict(X_tr_encoded)
    y_val_pred_log   = lr_model.predict(X_val_encoded)
    y_test_pred_log  = lr_model.predict(X_test_encoded)

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate_log(y_true, y_pred_log, label="Set"):
        if log_transform:
            y_pred_inv = np.expm1(y_pred_log)
        else:
            y_pred_inv = y_pred_log
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_inv))
        r2 = r2_score(y_true, y_pred_inv)
        print(f"{label} → RMSE: {rmse:.2f}, R²: {r2:.4f}")

    evaluate_log(y_train, y_train_pred_log, "Train")
    evaluate_log(y_val, y_val_pred_log, "Validation")
    evaluate_log(y_test, y_test_pred_log, "Test")

    return lr_model, X_tr_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test
