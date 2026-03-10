import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from xgboost import XGBRegressor

# =========================================================
# 🔥 MASTER FUNCTION — Pipeline 1 (Basic XGBoost)
# =========================================================
def run_xgb_pipeline_basic(df):
    """
    Basit XGBoost pipeline
    """

    # -----------------------------
    # 0️⃣ Remove price <= 1 & drop unwanted columns
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
    # 1️⃣ Split train / val / test
    # -----------------------------
    X = df.drop(columns=['Price'])
    y = df['Price']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train_df = X_train.copy()
    X_val_df   = X_val.copy()
    X_test_df  = X_test.copy()

    # -----------------------------
    # 2️⃣ Missing value handling
    # -----------------------------
    zero_cols = ['Garage', 'Number of garages', 'Swimming pool', 'Terrace', 
                 'Elevator', 'Garden']

    for col in zero_cols:
        X_train_df[col] = X_train_df[col].fillna(0)
        X_val_df[col] = X_val_df[col].fillna(0)
        X_test_df[col] = X_test_df[col].fillna(0)

    categorical_cols = ['Type of heating', 'State of the property']
    for col in categorical_cols:
        X_train_df[col] = X_train_df[col].fillna('unknown')
        X_val_df[col] = X_val_df[col].fillna('unknown')
        X_test_df[col] = X_test_df[col].fillna('unknown')

    # main_type
    for df_ in [X_train_df, X_val_df, X_test_df]:
        df_['main_type'] = df_['main_type'].fillna('unknown')

    # Special handling for Total land surface
    X_train_df.loc[X_train_df['main_type']=='apartment', 'Total land surface'] = \
        X_train_df.loc[X_train_df['main_type']=='apartment', 'Total land surface'].fillna(0)

    X_val_df.loc[X_val_df['main_type']=='apartment', 'Total land surface'] = \
        X_val_df.loc[X_val_df['main_type']=='apartment', 'Total land surface'].fillna(0)

    X_test_df.loc[X_test_df['main_type']=='apartment', 'Total land surface'] = \
        X_test_df.loc[X_test_df['main_type']=='apartment', 'Total land surface'].fillna(0)

    house_median = X_train_df.loc[X_train_df['main_type']=='house', 'Total land surface'].median()
    land_median  = X_train_df.loc[X_train_df['main_type']=='land', 'Total land surface'].median()

    for df_ in [X_train_df, X_val_df, X_test_df]:
        df_.loc[(df_['main_type']=='house') & (df_['Total land surface'].isna()), 'Total land surface'] = house_median
        df_.loc[(df_['main_type']=='land') & (df_['Total land surface'].isna()), 'Total land surface'] = land_median

    # Median fill
    num_cols_for_median = ['Number of bedrooms', 'Livable surface', 'Total land surface']
    for col in num_cols_for_median:
        median_dict = X_train_df.groupby('main_type')[col].median()

        for mtype, med_val in median_dict.items():
            X_train_df.loc[(X_train_df['main_type']==mtype) & (X_train_df[col].isna()), col] = med_val
            X_val_df.loc[(X_val_df['main_type']==mtype) & (X_val_df[col].isna()), col] = med_val
            X_test_df.loc[(X_test_df['main_type']==mtype) & (X_test_df[col].isna()), col] = med_val

        global_median = X_train_df[col].median()
        X_train_df[col] = X_train_df[col].fillna(global_median)
        X_val_df[col] = X_val_df[col].fillna(global_median)
        X_test_df[col] = X_test_df[col].fillna(global_median)

    # -----------------------------
    # 3️⃣ Outlier removal
    # -----------------------------
    outlier_cols = ['Number of bedrooms', 'Livable surface', 'Garage', 'Number of garages', 
                    'Terrace', 'Total land surface', 'Swimming pool']

    y_train = y_train.reindex(X_train_df.index)
    clean_blocks = []

    for mtype, group in X_train_df.groupby('main_type'):
        group_copy = group.copy()
        group_copy['Price'] = y_train.loc[group_copy.index]
        z = np.abs(stats.zscore(group_copy[outlier_cols], nan_policy='omit'))
        mask = (z < 3).all(axis=1)
        clean_blocks.append(group_copy[mask])

    df_clean_train = pd.concat(clean_blocks)
    X_tr = df_clean_train.drop(columns=['Price'])
    y_tr = df_clean_train['Price']

    X_val_copy = X_val_df.copy()
    X_test_copy = X_test_df.copy()

    # -----------------------------
    # 4️⃣ Feature Engineering
    # -----------------------------
    for df_ in [X_tr, X_val_copy, X_test_copy]:
        df_['has_swimming_pool'] = (df_['Swimming pool'] > 0).astype(int)
        df_['has_garden'] = (df_['Garden'] > 0).astype(int)
        df_['has_terrace'] = (df_['Terrace'] > 0).astype(int)
        df_['surface_ratio'] = df_['Livable surface'] / df_['Total land surface'].replace(0, 1)
        df_['area_per_bedroom'] = df_['Livable surface'] / df_['Number of bedrooms'].replace(0, 1)

    # -----------------------------
    # 5️⃣ Encoding
    # -----------------------------
    cat_cols = ['State of the property','Type of heating','type','city','Region','province','main_type']

    rare_thresh = 10
    for df_ in [X_tr, X_val_copy, X_test_copy]:
        city_counts = df_['city'].value_counts()
        rare = city_counts[city_counts < rare_thresh].index
        df_['city'] = df_['city'].replace(rare, 'other')

    existing_cat = [c for c in cat_cols if c in X_tr.columns]

    X_tr_enc = pd.get_dummies(X_tr, columns=existing_cat, drop_first=True)
    X_val_enc = pd.get_dummies(X_val_copy, columns=existing_cat, drop_first=True)
    X_test_enc = pd.get_dummies(X_test_copy, columns=existing_cat, drop_first=True)

    # Align
    X_val_enc = X_val_enc.reindex(columns=X_tr_enc.columns, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=X_tr_enc.columns, fill_value=0)

    # -----------------------------
    # 6️⃣ Train model
    # -----------------------------
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr_enc, y_tr)

    return model, X_tr_enc, X_val_enc, X_test_enc, y_tr, y_val, y_test


# =========================================================
# 🔥 Plot Function
# =========================================================
def plot_xgb_pipeline_results(model, X_tr, X_val, X_test, y_tr, y_val, y_test):
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_train_pred = model.predict(X_tr)
    y_val_pred   = model.predict(X_val)
    y_test_pred  = model.predict(X_test)

    plt.figure(figsize=(18,5))

    # Train
    plt.subplot(1,3,1)
    plt.scatter(y_tr, y_train_pred, alpha=0.4)
    plt.plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], 'r--')
    plt.title("Train: Actual vs Predicted")

    # Val
    plt.subplot(1,3,2)
    plt.scatter(y_val, y_val_pred, alpha=0.4)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title("Validation: Actual vs Predicted")

    # Test
    plt.subplot(1,3,3)
    plt.scatter(y_test, y_test_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Test: Actual vs Predicted")

    plt.tight_layout()
    plt.show()
