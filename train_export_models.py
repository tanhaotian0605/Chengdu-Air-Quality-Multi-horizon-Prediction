# -*- coding: utf-8 -*-
"""train_export_models_v2_no_leakage.py

修复点（很关键）：
- 你当前导出的 bundle['feature_cols'] 里包含了其它步长的目标列（例如 h=6 的模型把 AQI_t+1 当作特征），
  这会导致：
  1) 手动输入特征时无法提供这些“未来真实值”，从而报 Missing features；
  2) 这属于典型的数据泄露（target leakage），线上预测不可用。

因此这里做的唯一逻辑修正是：
- 训练每个 horizon 时，特征列统一排除所有 'AQI_t+' 开头的列（所有多步长目标列都不作为特征）。

除此之外：数据读取、城市聚合、插值/裁剪、时间特征、lag特征、模型参数、stacking方式均保持不变。

输出：
    models/model_ensemble_h1.joblib
    models/model_ensemble_h6.joblib
    models/model_ensemble_h12.joblib

运行：
    python train_export_models_v2_no_leakage.py --data_path "你的CSV路径" --model_dir "models"
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
import joblib


# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_raw_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")

    if "datetime" not in df.columns:
        raise ValueError("Column 'datetime' not found.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df


def build_city_level(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    city_df = df.groupby("datetime")[num_cols].mean(numeric_only=True).reset_index()
    return city_df


def preprocess_city_for_model(city_df: pd.DataFrame) -> pd.DataFrame:
    df = city_df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")

    pollutant_cols = [
        "AQI",
        "CO", "NO2", "O3", "O3_8h",
        "PM10", "PM2.5", "SO2",
        "CO_24h", "NO2_24h", "O3_24h", "O3_8h_24h",
        "PM10_24h", "PM2.5_24h", "SO2_24h"
    ]
    cols = [c for c in pollutant_cols if c in df.columns]
    for col in cols:
        q_low = df[col].quantile(0.001)
        q_hi = df[col].quantile(0.999)
        lower = max(0, q_low)
        df[col] = df[col].clip(lower, q_hi)
    return df


# -----------------------------
# Feature engineering
# -----------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime")
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "AQI") -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3, 6, 12]:
        df[f"AQI_lag{lag}"] = df[target_col].shift(lag)

    pollutant_cols = [
        "CO", "NO2", "O3", "O3_8h",
        "PM10", "PM2.5", "SO2",
        "CO_24h", "NO2_24h", "O3_24h", "O3_8h_24h",
        "PM10_24h", "PM2.5_24h", "SO2_24h"
    ]
    for col in pollutant_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    meteo_cols = [
        "Temperature", "Humidity", "WindSpeed",
        "WindDir", "Pressure", "BLH", "CloudCover"
    ]
    for col in meteo_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    return df


def add_multi_step_targets(df: pd.DataFrame, target_col: str = "AQI", horizons=(1, 6, 12)) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"AQI_t+{h}"] = df[target_col].shift(-h)
    return df


def build_supervised_dataset(city_df_model: pd.DataFrame, target_col: str = "AQI", horizons=(1, 6, 12)) -> pd.DataFrame:
    df_ts = city_df_model.copy()
    df_ts = add_time_features(df_ts)
    df_ts = add_lag_features(df_ts, target_col=target_col)
    df_ts = add_multi_step_targets(df_ts, target_col=target_col, horizons=horizons)
    df_ts = df_ts.dropna().reset_index(drop=True)
    return df_ts


def time_based_split(df_ts: pd.DataFrame, target_col: str, train_ratio=0.70, valid_ratio=0.15):
    df_ts = df_ts.sort_values("datetime")
    t1 = df_ts["datetime"].quantile(train_ratio)
    t2 = df_ts["datetime"].quantile(train_ratio + valid_ratio)

    train_df = df_ts[df_ts["datetime"] <= t1].copy()
    valid_df = df_ts[(df_ts["datetime"] > t1) & (df_ts["datetime"] <= t2)].copy()
    test_df = df_ts[df_ts["datetime"] > t2].copy()

    # 关键修正：排除所有 AQI_t+* 目标列（避免其它步长目标进入特征）
    feature_cols = [c for c in df_ts.columns if c != "datetime" and not c.startswith("AQI_t+")]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_valid, y_valid = valid_df[feature_cols], valid_df[target_col]
    X_test,  y_test  = test_df[feature_cols],  test_df[target_col]

    return (X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols)


# -----------------------------
# Modeling
# -----------------------------
def evaluate_model(name: str, phase: str, horizon: int, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "phase": phase, "horizon": horizon, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_predict_ridge(X_train, y_train, X_valid, y_valid, X_test, y_test, horizon: int):
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    ridge_pipe.fit(X_train, y_train)
    return ridge_pipe, ridge_pipe.predict(X_valid), ridge_pipe.predict(X_test)


def train_predict_rf(X_train, y_train, X_valid, y_valid, X_test, y_test, horizon: int):
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        min_samples_split=50, min_samples_leaf=20,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    return rf, rf.predict(X_valid), rf.predict(X_test)


def train_predict_xgb(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols, horizon: int):
    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return xgb, xgb.predict(X_valid), xgb.predict(X_test)


def train_predict_svr(X_train, y_train, X_valid, y_valid, X_test, y_test, horizon: int, max_sample: int = 100_000):
    if len(X_train) > max_sample:
        rs = np.random.RandomState(42)
        idx = rs.choice(len(X_train), max_sample, replace=False)
        X_train_svr, y_train_svr = X_train.iloc[idx], y_train.iloc[idx]
    else:
        X_train_svr, y_train_svr = X_train, y_train

    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, epsilon=0.1))
    ])
    svr_pipe.fit(X_train_svr, y_train_svr)
    return svr_pipe, svr_pipe.predict(X_valid), svr_pipe.predict(X_test)


def train_predict_ensemble(base_valid_preds: dict, base_test_preds: dict, y_valid, y_test, horizon: int, base_model_names=None):
    if base_model_names is None:
        base_model_names = ["Ridge", "RF", "XGB", "SVR"]
    stack_X_valid = np.vstack([base_valid_preds[m] for m in base_model_names]).T
    stack_X_test = np.vstack([base_test_preds[m] for m in base_model_names]).T

    meta = Ridge(alpha=1.0, random_state=42)
    meta.fit(stack_X_valid, y_valid)
    return meta, meta.predict(stack_X_valid), meta.predict(stack_X_test)


# -----------------------------
# Main
# -----------------------------
def main(data_path: str, model_dir: str, horizons=(1, 6, 12), svr_max_sample: int = 100_000):
    os.makedirs(model_dir, exist_ok=True)

    df_raw = load_raw_data(data_path)
    city_df = preprocess_city_for_model(build_city_level(df_raw))
    df_ts = build_supervised_dataset(city_df, target_col="AQI", horizons=horizons)

    for h in horizons:
        target_col = f"AQI_t+{h}"
        X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols = time_based_split(df_ts, target_col=target_col)

        ridge_model, ridge_valid, ridge_test = train_predict_ridge(X_train, y_train, X_valid, y_valid, X_test, y_test, h)
        rf_model, rf_valid, rf_test = train_predict_rf(X_train, y_train, X_valid, y_valid, X_test, y_test, h)
        xgb_model, xgb_valid, xgb_test = train_predict_xgb(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols, h)
        svr_model, svr_valid, svr_test = train_predict_svr(X_train, y_train, X_valid, y_valid, X_test, y_test, h, max_sample=svr_max_sample)

        base_model_names = ["Ridge", "RF", "XGB", "SVR"]
        meta_model, _, _ = train_predict_ensemble(
            base_valid_preds={"Ridge": ridge_valid, "RF": rf_valid, "XGB": xgb_valid, "SVR": svr_valid},
            base_test_preds={"Ridge": ridge_test,  "RF": rf_test,  "XGB": xgb_test,  "SVR": svr_test},
            y_valid=y_valid, y_test=y_test, horizon=h, base_model_names=base_model_names
        )

        bundle = {
            "horizon": h,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "base_model_names": base_model_names,
            "base_models": {"Ridge": ridge_model, "RF": rf_model, "XGB": xgb_model, "SVR": svr_model},
            "meta_model": meta_model
        }

        out_file = os.path.join(model_dir, f"model_ensemble_h{h}.joblib")
        joblib.dump(bundle, out_file)
        print("Saved:", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to merged dataset CSV.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models.")
    parser.add_argument("--svr_max_sample", type=int, default=100_000, help="Max sample size for SVR.")
    args = parser.parse_args()

    main(args.data_path, args.model_dir, horizons=(1, 6, 12), svr_max_sample=args.svr_max_sample)
