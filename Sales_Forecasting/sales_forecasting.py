import warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose

import lightgbm as lgb
import xgboost as xgb

# ------------------ CONFIG ------------------ #
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
FEATURES_PATH = "features.csv"
STORES_PATH  = "stores.csv"

STORE_ID = None
DEPT_ID = None

RANDOM_STATE = 42
N_SPLITS  = 5
FORECAST_WEEKS = 12

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ------------------ UTILS ------------------ #
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def time_features(df, date_col="Date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["dayofyear"] = df[date_col].dt.dayofyear
    df["quarter"] = df[date_col].dt.quarter
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df[date_col].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df[date_col].dt.is_quarter_end.astype(int)
    df["is_year_start"] = df[date_col].dt.is_year_start.astype(int)
    df["is_year_end"] = df[date_col].dt.is_year_end.astype(int)
    
    # Holiday features (US holidays)
    df['is_christmas'] = ((df['month'] == 12) & (df['dayofyear'] >= 355)).astype(int)
    df['is_thanksgiving'] = ((df['month'] == 11) & (df['dayofyear'] >= 330) & (df['dayofyear'] <= 336)).astype(int)
    df['is_superbowl'] = ((df['month'] == 2) & (df['dayofyear'] >= 40) & (df['dayofyear'] <= 46)).astype(int)
    
    return df

def add_group_lags_rolls(df, group_cols, target_col, lags=(1,2,3,4,5,12,52), roll_windows=(4,8,12,26)):
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"])
    
    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    
    # Rolling statistics
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df.groupby(group_cols)[target_col].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df.groupby(group_cols)[target_col].shift(1).rolling(w, min_periods=1).std().fillna(0)
        df[f"roll_min_{w}"] = df.groupby(group_cols)[target_col].shift(1).rolling(w, min_periods=1).min()
        df[f"roll_max_{w}"] = df.groupby(group_cols)[target_col].shift(1).rolling(w, min_periods=1).max()
    
    # Percentage changes
    df["pct_change_1"] = df.groupby(group_cols)[target_col].pct_change(periods=1).fillna(0)
    df["pct_change_4"] = df.groupby(group_cols)[target_col].pct_change(periods=4).fillna(0)
    df["pct_change_12"] = df.groupby(group_cols)[target_col].pct_change(periods=12).fillna(0)
    
    # Exponential moving averages
    df["ema_4"] = df.groupby(group_cols)[target_col].shift(1).ewm(span=4, adjust=False).mean()
    df["ema_12"] = df.groupby(group_cols)[target_col].shift(1).ewm(span=12, adjust=False).mean()
    
    return df

def preprocess_features_df(feat_df):
    feat_df = feat_df.copy()
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])
    
    # Handle missing values in markdown columns
    markdown_cols = [col for col in feat_df.columns if 'MarkDown' in col]
    for col in markdown_cols:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna(0.0)
    
    # Handle other missing values
    num_cols = feat_df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if col not in markdown_cols:  # Don't re-process markdown columns
            feat_df[col] = feat_df[col].fillna(feat_df[col].median())
    
    if "IsHoliday" in feat_df.columns:
        feat_df["IsHoliday"] = feat_df["IsHoliday"].astype(int)
    
    return feat_df

def merge_all(train_df, features_df, stores_df):
    df = train_df.merge(features_df, on=["Store","Date"], how="left")
    df = df.merge(stores_df, on="Store", how="left")
    return df

def choose_series(train_df):
    # Select the series with the most complete history
    grp = train_df.groupby(["Store","Dept"]).size().sort_values(ascending=False)
    store, dept = grp.index[0]
    return int(store), int(dept)

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'get_score'):
        # For XGBoost models
        importance = np.array([model.get_score().get(f, 0) for f in feature_names])
    else:
        print("Model doesn't have feature importance")
        return
    
    # Create feature importance dataframe
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp, x='importance', y='feature')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------------ LOAD DATA ------------------ #
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
features = pd.read_csv(FEATURES_PATH)
stores = pd.read_csv(STORES_PATH)
test = pd.read_csv(TEST_PATH)

# Convert dates to datetime
for df in [train, features, stores, test]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

# Preprocess features
features = preprocess_features_df(features)
train["IsHoliday"] = train["IsHoliday"].astype(int)
test["IsHoliday"] = test["IsHoliday"].astype(int)

# Merge all datasets
train_full = merge_all(train, features, stores)
test_full = merge_all(test, features, stores)

# Add time features
train_full = time_features(train_full, "Date")
test_full = time_features(test_full, "Date")

# Encode store types
le = LabelEncoder()
train_full['Type_encoded'] = le.fit_transform(train_full['Type'])
test_full['Type_encoded'] = le.transform(test_full['Type'])

# ------------------ SINGLE SERIES MODEL ------------------ #
if STORE_ID is None or DEPT_ID is None:
    STORE_ID, DEPT_ID = choose_series(train_full)

print(f"Selected series for analysis: Store={STORE_ID}, Dept={DEPT_ID}")

# Extract the time series for the selected store and department
ts = train_full[(train_full["Store"]==STORE_ID) & (train_full["Dept"]==DEPT_ID)].copy()
ts = ts.sort_values("Date")

# Add lag and rolling features
ts = add_group_lags_rolls(ts, ["Store","Dept"], "Weekly_Sales")

# Define features and target
drop_cols = ["Weekly_Sales","Date","Type"]
feature_cols = [c for c in ts.columns if c not in drop_cols and not re.match(r"^Unnamed", c)]

# Remove rows with missing values from lag features
ts_model = ts.dropna(subset=[c for c in ts.columns if c.startswith("lag_")]).reset_index(drop=True)
X = ts_model[feature_cols].copy()
y = ts_model["Weekly_Sales"].copy()
dates = ts_model["Date"].copy()

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ------------------ EXPLORATORY DATA ANALYSIS ------------------ #
# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(ts_model["Date"], ts_model["Weekly_Sales"])
plt.title(f"Weekly Sales - Store {STORE_ID}, Dept {DEPT_ID}")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Seasonal Decomposition
try:
    ts_agg = ts.set_index("Date")["Weekly_Sales"].asfreq("W-FRI").interpolate()
    decomp = seasonal_decompose(ts_agg, model="additive", period=52, extrapolate_trend='freq')
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle(f"Seasonal Decomposition - Store {STORE_ID}, Dept {DEPT_ID}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Decomposition failed. Error:", e)

# ------------------ CROSS VALIDATION ------------------ #
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
lgb_scores, xgb_scores = [], []
lgb_models, xgb_models = [], []

print("Starting cross-validation...")
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_scaled), start=1):
    X_tr, X_va = X_scaled.iloc[tr_idx], X_scaled.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': RANDOM_STATE,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1
    }
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    y_va_pred = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    lgb_scores.append(rmse(y_va, y_va_pred))
    lgb_models.append(lgb_model)

    # XGBoost - CORRECTED VERSION
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective='reg:squarederror',
        verbosity=0,
        early_stopping_rounds=100  # Moved to constructor
    )
    
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    
    y_va_pred_x = xgb_model.predict(X_va)
    xgb_scores.append(rmse(y_va, y_va_pred_x))
    xgb_models.append(xgb_model)
    
    print(f"Fold {fold}: LGB RMSE={lgb_scores[-1]:.3f} | XGB RMSE={xgb_scores[-1]:.3f}")

print("\nCV Summary:")
print(f"LightGBM RMSE: {np.mean(lgb_scores):.3f} ± {np.std(lgb_scores):.3f}")
print(f"XGBoost RMSE: {np.mean(xgb_scores):.3f} ± {np.std(xgb_scores):.3f}")

use_lgb = np.mean(lgb_scores) <= np.mean(xgb_scores)
model_name = "LightGBM" if use_lgb else "XGBoost"
best_model = lgb_models[np.argmin(lgb_scores)] if use_lgb else xgb_models[np.argmin(xgb_scores)]

# ------------------ HOLDOUT TEST ------------------ #
split_idx = int(len(X_scaled) * 0.8)
X_tr_full, X_ho = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
y_tr_full, y_ho = y.iloc[:split_idx], y.iloc[split_idx:]
dates_ho = dates.iloc[split_idx:]

if use_lgb:
    final_model = lgb.train(
        lgb_params, 
        lgb.Dataset(X_tr_full, label=y_tr_full),
        num_boost_round=best_model.best_iteration
    )
    y_ho_pred = final_model.predict(X_ho)
else:
    # CORRECTED: For XGBoost, use the best iteration from cross-validation
    final_model = xgb.XGBRegressor(
        n_estimators=best_model.best_iteration,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective='reg:squarederror',
        verbosity=0
    )
    final_model.fit(X_tr_full, y_tr_full)
    y_ho_pred = final_model.predict(X_ho)

# Calculate evaluation metrics
ho_rmse = rmse(y_ho, y_ho_pred)
ho_mae = mean_absolute_error(y_ho, y_ho_pred)
ho_mape = mape(y_ho, y_ho_pred)
ho_r2 = r2_score(y_ho, y_ho_pred)

print(f"\nChosen model: {model_name}")
print(f"Holdout RMSE: {ho_rmse:.3f}")
print(f"Holdout MAE: {ho_mae:.3f}")
print(f"Holdout MAPE: {ho_mape:.2f}%")
print(f"Holdout R2: {ho_r2:.3f}")

# Plot feature importance
plot_feature_importance(final_model, feature_cols, f"Feature Importance - {model_name}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(dates_ho, y_ho.values, marker='o', label='Actual', linewidth=2)
plt.plot(dates_ho, y_ho_pred, marker='x', label='Predicted', linewidth=2, linestyle='--')
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title(f"Actual vs Predicted (Holdout) - Store {STORE_ID}, Dept {DEPT_ID} - {model_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------ FORECAST FUTURE SALES ------------------ #
def forecast_future(model, last_data, scaler, feature_cols, periods=FORECAST_WEEKS):
    """Forecast future sales using the trained model"""
    forecasts = []
    current_data = last_data.copy()
    
    for i in range(periods):
        # Prepare features for prediction
        X_pred = current_data[feature_cols].values.reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        if use_lgb:
            pred = model.predict(X_pred_scaled)
        else:
            pred = model.predict(X_pred_scaled)
        
        forecasts.append(pred[0])
        
        # Update the data for next prediction
        # Shift lag features
        for lag in range(52, 1, -1):
            if f"lag_{lag}" in feature_cols:
                current_data[f"lag_{lag}"] = current_data.get(f"lag_{lag-1}", 0)
        
        # Update lag_1 with the current prediction
        current_data["lag_1"] = pred[0]
        
        # Update rolling statistics (simplified)
        for w in [4, 8, 12, 26]:
            if f"roll_mean_{w}" in feature_cols:
                # Simplified update - in practice would need to maintain a window
                current_data[f"roll_mean_{w}"] = current_data.get(f"roll_mean_{w}", 0) * 0.9 + pred[0] * 0.1
    
    return forecasts

# Get the most recent data point
last_data_point = ts.iloc[-1:].copy()
future_forecasts = forecast_future(final_model, last_data_point, scaler, feature_cols, FORECAST_WEEKS)

# Create future dates
last_date = ts["Date"].max()
future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, FORECAST_WEEKS+1)]

# Plot historical data and forecast
plt.figure(figsize=(14, 7))
plt.plot(ts["Date"], ts["Weekly_Sales"], label="Historical", linewidth=2)
plt.plot(future_dates, future_forecasts, label="Forecast", linewidth=2, linestyle='--', color='red')
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title(f"Sales Forecast - Store {STORE_ID}, Dept {DEPT_ID}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nFuture Forecast:")
for i, (date, forecast) in enumerate(zip(future_dates, future_forecasts), 1):
    print(f"Week {i} ({date.strftime('%Y-%m-%d')}): ${forecast:,.2f}")

# ------------------ GLOBAL MODEL FOR ALL STORES/DEPTS ------------------ #
print("\nTraining global model for all stores and departments...")

# Prepare global dataset
train_all = add_group_lags_rolls(train_full, ["Store","Dept"], "Weekly_Sales")
test_all = test_full.copy()

# Combine train and test to create consistent lag features
temp = pd.concat([
    train_all[["Store","Dept","Date","Weekly_Sales"]],
    test_all[["Store","Dept","Date"]].assign(Weekly_Sales=np.nan)
]).sort_values(["Store","Dept","Date"]).reset_index(drop=True)

# Add lag and rolling features to the combined dataset
temp = add_group_lags_rolls(temp, ["Store","Dept"], "Weekly_Sales")

# Split back into train and test
train_feats = temp[temp["Weekly_Sales"].notna()].merge(
    train_full.drop(columns=["Weekly_Sales"]), on=["Store","Dept","Date"], how="left")
train_feats["Weekly_Sales"] = temp.loc[temp["Weekly_Sales"].notna(), "Weekly_Sales"].values

test_feats = temp[temp["Weekly_Sales"].isna()].merge(
    test_full, on=["Store","Dept","Date"], how="left")

# Define features for global model
drop_cols_global = ["Weekly_Sales","Date","Type"]
feat_cols_global = [c for c in train_feats.columns if c not in drop_cols_global and not re.match(r"^Unnamed", c)]

Xg = train_feats[feat_cols_global].copy()
yg = train_feats["Weekly_Sales"].copy()
Xg_test = test_feats[feat_cols_global].copy()

# Scale features
scaler_g = StandardScaler()
Xg_scaled = pd.DataFrame(scaler_g.fit_transform(Xg), columns=Xg.columns)
Xg_test_scaled = pd.DataFrame(scaler_g.transform(Xg_test), columns=Xg_test.columns)

# Train final global model
print("Training global LightGBM model...")
lgb_train_g = lgb.Dataset(Xg_scaled, label=yg)
lgb_params_g = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'seed': RANDOM_STATE,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1
}

final_g = lgb.train(
    lgb_params_g, 
    lgb_train_g, 
    num_boost_round=1000,
    callbacks=[lgb.log_evaluation(200)]
)

test_preds = final_g.predict(Xg_test_scaled)

print("\nGlobal model training completed")

# Create submission file
submission = test[["Store","Dept","Date"]].copy()
submission["Weekly_Sales"] = test_preds
submission = submission.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
submission.to_csv("submission.csv", index=False)
print("✅ Saved predictions as submission.csv")

# Plot feature importance for global model
plot_feature_importance(final_g, feat_cols_global, "Global Model Feature Importance")

print("\nScript execution completed successfully!")