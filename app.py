# -*- coding: utf-8 -*-
"""
Chengdu Air Quality Multi-horizon Prediction (1h, 6h, 12h) — STRICT6

Aligned with exported strict6 bundles:
- Features come from bundle["feature_cols"] only (no hidden columns).
- No O3_8h, no *_24h, no contemporaneous rolling leakage.
- WindDir is NOT a feature; use wind_x / wind_y (and lag1).

Run:
    streamlit run app.py
"""

import os
import re
import math
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config + minimal CSS
# -----------------------------
st.set_page_config(
    page_title="Chengdu Air Quality Multi-horizon Prediction (STRICT6)",
    page_icon="🌫️",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      .hint { opacity: .85; font-size: 0.95rem; }
      .pill { display:inline-block; padding: 3px 10px; border-radius: 999px;
              border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04); margin-right: 6px; }
      div[data-testid="stForm"] { border-radius: 16px; border: 1px solid rgba(255,255,255,.08); padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Load exported model bundles (STRICT6) - LAZY
#   ✅ Default load ONLY h=1 at startup
#   ✅ Load h=6/h=12 on demand when predicting
# -----------------------------
def _load_bundle(rel_path: str):
    p = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Cannot find model file: {p}")
    return joblib.load(p)


@st.cache_resource
def load_bundle(h: int) -> dict:
    if h not in (1, 6, 12):
        raise ValueError(f"Unsupported horizon: {h}. Expected one of (1,6,12).")
    rel = f"exports/models_strict6_compressed/ensemble_bundle_h{h}.joblib.xz"
    return _load_bundle(rel)


# -----------------------------
# Built-in defaults
# NOTE: You can keep your old baked defaults as a base.
# We'll auto-adapt them to STRICT6 feature_cols:
# - ignore O3_8h / *_24h if present
# - compute wind_x/wind_y from WindSpeed + WindDir if needed
# - fill missing lag features using lag1/current values to avoid zeros
# -----------------------------
EMBEDDED_LATEST_DT = pd.Timestamp("2025-11-27 07:00:00")

# Your existing baked defaults (old). It's OK if it contains extra keys;
# we will filter it down to model feature_cols.
EMBEDDED_DEFAULTS_RAW = {
    'AQI': 137.625,
    'AQI_lag1': 139.875,
    'AQI_lag2': 124.0,
    'AQI_lag3': 121.75,
    'AQI_lag6': 124.0,
    'AQI_lag12': 72.875,
    'AQI_lag24': 0.0,  # if missing previously, keep 0; we will backfill below
    'CO': 0.9625,
    'CO_lag1': 0.9,
    'NO2': 41.25,
    'NO2_lag1': 42.875,
    'O3': 5.166666666666667,
    'O3_lag1': 6.333333333333333,
    'PM10': 130.75,
    'PM10_lag1': 126.75,
    'PM2.5': 104.375,
    'PM2.5_lag1': 106.0,
    'SO2': 6.875,
    'SO2_lag1': 5.375,

    # meteorology (old)
    'Temperature': 17.09625,
    'Temperature_lag1': 16.4225,
    'Humidity': 49.175,
    'Humidity_lag1': 50.825,
    'WindSpeed': 2.10875,
    'WindSpeed_lag1': 2.08125,
    'WindDir': 129.25,
    'WindDir_lag1': 95.75,
    'Pressure': 961.45,
    'Pressure_lag1': 962.8875,
    'BLH': 749.7,
    'BLH_lag1': 720.425,
    'CloudCover': 0.0025,
    'CloudCover_lag1': 0.0,

    # location/time
    'latitude': 30.6463375,
    'longitude': 104.0793,
    'hour': 7.0,
    'dayofweek': 3.0,
    'month': 11.0,
    'is_weekend': 0.0,

    # legacy keys (will be ignored anyway)
    'O3_8h': 53.0,
    'O3_8h_lag1': 53.0,
    'CO_24h': 0.85,
    'NO2_24h': 49.0,
}


STRICT_POLLUTANTS = ["CO", "NO2", "O3", "PM10", "PM2.5", "SO2"]
STRICT_POLLUTANT_LAGS = [1, 3, 6, 12, 24]   # per your strict plan


def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def build_defaults_for_features(feature_cols: list[str], base: dict) -> dict:
    """
    Create a defaults dict that matches exactly feature_cols, without forbidden derived columns.
    Also compute wind_x/wind_y from WindSpeed+WindDir if required.
    """
    base = dict(base)  # copy

    # 1) Remove forbidden stuff from base (safe even if not present)
    base = {k: v for k, v in base.items() if ("_24h" not in k and "O3_8h" not in k and not k.startswith("AQI_t+"))}

    # 2) Compute wind_x/wind_y (and lag1) if needed by the model
    need_wind_xy = any(k in feature_cols for k in ["wind_x", "wind_y", "wind_x_lag1", "wind_y_lag1"])
    if need_wind_xy:
        ws = float(base.get("WindSpeed", 0.0))
        wd = float(base.get("WindDir", 0.0))
        base["wind_x"] = ws * math.cos(_deg2rad(wd))
        base["wind_y"] = ws * math.sin(_deg2rad(wd))

        ws1 = float(base.get("WindSpeed_lag1", ws))
        wd1 = float(base.get("WindDir_lag1", wd))
        base["wind_x_lag1"] = ws1 * math.cos(_deg2rad(wd1))
        base["wind_y_lag1"] = ws1 * math.sin(_deg2rad(wd1))

    # 3) Backfill common missing lag features using lag1 or current
    #    e.g., CO_lag3 missing -> use CO_lag1 else CO
    lag_re = re.compile(r"^(?P<base>.+)_lag(?P<n>\d+)$")

    def _get_backfill(feat: str) -> float:
        # direct
        if feat in base:
            return float(base[feat])

        m = lag_re.match(feat)
        if m:
            b = m.group("base")
            # try lag1
            if f"{b}_lag1" in base:
                return float(base[f"{b}_lag1"])
            # try current
            if b in base:
                return float(base[b])
            return 0.0

        # AQI rolling mean naming variants (best-effort)
        if feat.lower().startswith("aqi") and ("roll" in feat.lower() or "rm" in feat.lower() or "rolling" in feat.lower()):
            return float(base.get("AQI_lag1", base.get("AQI", 0.0)))

        # anything else
        return float(base.get(feat, 0.0))

    out = {f: _get_backfill(f) for f in feature_cols}
    return out


# -----------------------------
# Inference
# -----------------------------
def predict_with_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """
    For your exported strict6 bundle structure:
      - bundle['feature_cols']
      - bundle['models'] = {'ridge','rf','svr','xgb'(maybe None),'meta'}
      - bundle['meta_feature_names'] defines the stacking order
    """
    feature_cols = bundle["feature_cols"]
    X_use = X[feature_cols]

    models = bundle["models"]
    meta = models["meta"]

    meta_feature_names = bundle.get("meta_feature_names", [])

    def _name_to_model_key(n: str) -> str | None:
        nl = str(n).lower()
        # 1) direct match
        if n in models:
            return n
        # 2) fuzzy match
        if "ridge" in nl:
            return "ridge"
        if nl in ("rf", "randomforest", "random_forest") or ("rf" in nl) or ("random" in nl and "forest" in nl):
            return "rf"
        if "svr" in nl:
            return "svr"
        if "xgb" in nl or "xgboost" in nl:
            return "xgb"
        return None

    parts = []

    # Preferred: follow meta_feature_names order
    for nm in meta_feature_names:
        k = _name_to_model_key(nm)
        if k is None:
            continue
        m = models.get(k, None)
        if m is None:
            continue
        parts.append(m.predict(X_use).reshape(-1, 1))

    # Fallback: if meta_feature_names not usable, use fixed order
    if not parts:
        for k in ["ridge", "rf", "svr", "xgb"]:
            m = models.get(k, None)
            if m is None:
                continue
            parts.append(m.predict(X_use).reshape(-1, 1))

    if not parts:
        raise KeyError(
            "No base model predictions available. "
            "Check bundle['models'] contains trained estimators."
        )

    meta_X = np.hstack(parts)
    return meta.predict(meta_X)


# -----------------------------
# Load ONLY h=1 model (need feature_cols early)
# -----------------------------
try:
    bundle_h1 = load_bundle(1)  # ✅ default load ONLY h=1
except Exception as e:
    st.error("❌ Failed to load STRICT6 model (h=1). Please check exports/models_strict6_compressed/ensemble_bundle_h1.joblib.xz")
    st.exception(e)
    st.stop()

feature_cols = bundle_h1["feature_cols"]


# -----------------------------
# Session state init (now that we know feature_cols)
# -----------------------------
STRICT_DEFAULTS = build_defaults_for_features(feature_cols, EMBEDDED_DEFAULTS_RAW)

if "defaults" not in st.session_state:
    st.session_state["defaults"] = STRICT_DEFAULTS.copy()

if "latest_dt" not in st.session_state:
    st.session_state["latest_dt"] = EMBEDDED_LATEST_DT

if "last_pred_df" not in st.session_state:
    st.session_state["last_pred_df"] = None

if "view" not in st.session_state:
    st.session_state["view"] = "Input"

if "manual_dt" not in st.session_state:
    st.session_state["manual_dt"] = EMBEDDED_LATEST_DT


# -----------------------------
# Header
# -----------------------------
st.title("Chengdu Air Quality Multi-horizon Prediction (STRICT6)")
st.markdown(
    '<span class="pill">Manual Input</span>'
    '<span class="pill">Built-in defaults (auto-adapted)</span>'
    '<span class="pill">Datetime → Time Features</span>'
    '<span class="pill">1/6/12h</span>',
    unsafe_allow_html=True
)


# -----------------------------
# Force defaults on version bump
# -----------------------------
APP_VERSION = "strict6_exports_v1"
if st.session_state.get("_app_version") != APP_VERSION:
    st.session_state["_app_version"] = APP_VERSION
    st.session_state["defaults"] = STRICT_DEFAULTS.copy()
    st.session_state["latest_dt"] = EMBEDDED_LATEST_DT
    st.session_state["manual_dt"] = EMBEDDED_LATEST_DT
    st.session_state["last_pred_df"] = None

    for feat in feature_cols:
        st.session_state[f"feat__{feat}"] = float(st.session_state["defaults"].get(feat, 0.0))


# -----------------------------
# Initialize inputs
# -----------------------------
for feat in feature_cols:
    k = f"feat__{feat}"
    if k not in st.session_state:
        st.session_state[k] = float(st.session_state["defaults"].get(feat, 0.0))


def _apply_time_from_manual_dt():
    md = st.session_state["manual_dt"]
    if "hour" in feature_cols:
        st.session_state["feat__hour"] = int(md.hour)
    if "dayofweek" in feature_cols:
        st.session_state["feat__dayofweek"] = int(md.dayofweek)
    if "month" in feature_cols:
        st.session_state["feat__month"] = int(md.month)
    if "is_weekend" in feature_cols:
        st.session_state["feat__is_weekend"] = int(md.dayofweek >= 5)


_apply_time_from_manual_dt()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.success("Models loaded ✅")  # (h=6/h=12 will be loaded on-demand)

    horizons = st.multiselect("⏱️ Forecast horizons", [1, 6, 12], default=[1])

    st.divider()
    st.subheader("🗓️ datetime_input (select date & time)")

    md0 = st.session_state["manual_dt"]
    init_date = md0.date()
    init_hour = int(md0.hour)

    pick_date = st.date_input("Date", value=init_date)
    hour_options = list(range(24))
    picked_hour = st.selectbox("Time (hour)", hour_options, index=init_hour, format_func=lambda h: f"{h:02d}:00")

    picked_dt = pd.Timestamp(dt.datetime.combine(pick_date, dt.time(hour=int(picked_hour), minute=0, second=0)))
    st.caption(f"Selected datetime: **{picked_dt.strftime('%Y-%m-%d %H:%M:%S')}**")

    if st.button("⚡ Update time features (hour/dayofweek/month/is_weekend)", use_container_width=True):
        st.session_state["manual_dt"] = picked_dt
        _apply_time_from_manual_dt()
        st.toast("Time features updated ✅", icon="🕒")

    st.divider()
    st.subheader("📌 Built-in defaults")

    if st.button("Reset to built-in defaults", use_container_width=True):
        st.session_state["defaults"] = build_defaults_for_features(feature_cols, EMBEDDED_DEFAULTS_RAW)
        st.session_state["latest_dt"] = EMBEDDED_LATEST_DT

        for feat in feature_cols:
            st.session_state[f"feat__{feat}"] = float(st.session_state["defaults"].get(feat, 0.0))

        _apply_time_from_manual_dt()
        st.toast("Defaults applied ✅", icon="✅")

    st.divider()
    st.subheader("ℹ️ Info")
    st.write(f"Built-in defaults timestamp: **{st.session_state['latest_dt']}**")
    st.write(f"Manual datetime: **{st.session_state['manual_dt']}**")
    st.write(f"Features required: **{len(feature_cols)}**")


# -----------------------------
# Feature grouping (STRICT6-aware)
# -----------------------------
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def group_features(cols: list[str]) -> dict[str, list[str]]:
    groups = {
        "🧭 Core (AQI)": [],
        "🕒 Time (locked)": [],
        "📉 AQI lags": [],
        "🧪 Pollutants (current)": [],
        "🧪 Pollutants (lags)": [],
        "🌤️ Meteorology (current)": [],
        "🌤️ Meteorology (lag1)": [],
        "📦 Others": [],
    }

    meteo_current = {"Temperature", "Humidity", "WindSpeed", "Pressure", "BLH", "CloudCover", "wind_x", "wind_y"}
    meteo_lag1 = {f"{m}_lag1" for m in meteo_current}

    for c in cols:
        if c == "AQI":
            groups["🧭 Core (AQI)"].append(c)
        elif c in {"longitude", "latitude"}:
            continue
        elif c in {"hour", "dayofweek", "month", "is_weekend"}:
            groups["🕒 Time (locked)"].append(c)
        elif c.startswith("AQI_lag"):
            groups["📉 AQI lags"].append(c)
        elif c in STRICT_POLLUTANTS:
            groups["🧪 Pollutants (current)"].append(c)
        elif any(c.startswith(p + "_lag") for p in STRICT_POLLUTANTS):
            groups["🧪 Pollutants (lags)"].append(c)
        elif c in meteo_current:
            groups["🌤️ Meteorology (current)"].append(c)
        elif c in meteo_lag1:
            groups["🌤️ Meteorology (lag1)"].append(c)
        else:
            groups["📦 Others"].append(c)

    # keep only non-empty groups
    return {k: v for k, v in groups.items() if v}


groups = group_features(feature_cols)


# -----------------------------
# Widgets (Time features locked)
# -----------------------------
def render_widget(container, feat: str):
    key = f"feat__{feat}"

    if feat == "hour":
        container.selectbox("hour (0-23)", list(range(24)), key=key, disabled=True)
        return

    if feat == "dayofweek":
        container.selectbox(
            "dayofweek",
            list(range(7)),
            format_func=lambda x: f"{x} ({DOW_LABELS[x]})",
            key=key,
            disabled=True
        )
        return

    if feat == "month":
        container.selectbox("month", list(range(1, 13)), key=key, disabled=True)
        return

    if feat == "is_weekend":
        container.selectbox("is_weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=key, disabled=True)
        return

    container.number_input(feat, key=key, format="%.6f")


# -----------------------------
# Page navigation
# -----------------------------
if st.session_state["view"] == "Input":
    st.subheader("🧾 Feature Input")

    c1, c2 = st.columns([2, 1])
    with c1:
        visible_groups = st.multiselect("Show groups", list(groups.keys()), default=list(groups.keys()))
    with c2:
        cols_per_row = st.select_slider("Inputs per row", [2, 3, 4], value=4)

    with st.form("feature_form"):
        for gname in visible_groups:
            with st.expander(gname, expanded=(gname in ["🧭 Core (AQI)", "🕒 Time (locked)", "📉 AQI lags"])):
                feats = groups[gname]
                ui_cols = st.columns(cols_per_row)
                for i, feat in enumerate(feats):
                    render_widget(ui_cols[i % cols_per_row], feat)

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        X = pd.DataFrame([{feat: float(st.session_state[f"feat__{feat}"]) for feat in feature_cols}])

        outputs = []
        for h in horizons:
            # ✅ On-demand load for h=6/h=12; h=1 is already cached
            y = float(predict_with_bundle(load_bundle(h), X)[0])
            outputs.append({"horizon_hours": h, "predicted_AQI": y})

        pred_df = pd.DataFrame(outputs).sort_values("horizon_hours").reset_index(drop=True)
        st.session_state["last_pred_df"] = pred_df

        st.session_state["view"] = "Results"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

else:
    st.subheader("📈 Prediction Results")

    if st.button("⬅ Back to Input", use_container_width=False):
        st.session_state["view"] = "Input"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    pred_df = st.session_state.get("last_pred_df")
    if pred_df is None or pred_df.empty:
        st.info("No prediction yet. Go to Input and click Predict.")
    else:
        def aqi_category(aqi_value: float) -> str:
            if aqi_value <= 50:
                return "Good"
            if aqi_value <= 100:
                return "Moderate"
            if aqi_value <= 150:
                return "Unhealthy for Sensitive Groups"
            if aqi_value <= 200:
                return "Unhealthy"
            if aqi_value <= 300:
                return "Very Unhealthy"
            return "Hazardous"

        baseline_aqi = None
        try:
            baseline_aqi = float(st.session_state.get("feat__AQI", np.nan))
            if np.isnan(baseline_aqi):
                baseline_aqi = None
        except Exception:
            baseline_aqi = None

        m1, m2, m3 = st.columns(3)
        for col, h in zip([m1, m2, m3], [1, 6, 12]):
            if (pred_df["horizon_hours"] == h).any():
                v = float(pred_df.loc[pred_df["horizon_hours"] == h, "predicted_AQI"].iloc[0])

                if baseline_aqi is None:
                    col.metric(f"t+{h}h AQI", f"{v:.2f}")
                else:
                    delta = v - baseline_aqi
                    col.metric(f"t+{h}h AQI", f"{v:.2f}", delta=f"{delta:+.2f}", delta_color="normal")

                col.markdown(
                    f"<div style='font-size:1.05rem; font-weight:650; margin-top:0.25rem;'>"
                    f"Level: {aqi_category(v)}</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.write("**Forecast curve**")
        st.line_chart(pred_df.set_index("horizon_hours"))
