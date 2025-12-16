# -*- coding: utf-8 -*-
"""
Chengdu Air Quality Multi-horizon Prediction (1h, 6h, 12h)

Notes:
- Built-in defaults are embedded (no runtime reading of city_hourly.parquet).
- datetime_input updates hour/dayofweek/month/is_weekend with one click.
- These 4 time features are locked in the input form and can only be updated via datetime_input.

Run:
    streamlit run app.py
"""


import os
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config + minimal CSS
# -----------------------------
st.set_page_config(page_title="Chengdu Air Quality Multi-horizon Prediction (1h, 6h, 12h)", page_icon="🌫️", layout="wide")

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
# Load exported model bundles
# -----------------------------
def _load_bundle(rel_path: str):
    p = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Cannot find model file: {p}")
    return joblib.load(p)


@st.cache_resource
def load_all_bundles():
    return {
        1: _load_bundle("models/model_ensemble_h1.joblib"),
        6: _load_bundle("models/model_ensemble_h6.joblib"),
        12: _load_bundle("models/model_ensemble_h12.joblib"),
    }


# -----------------------------
# Built-in defaults (baked from city_hourly.parquet latest valid row)
# NOTE: This avoids reading any data file at runtime.
# Latest datetime used: 2025-11-27 07:00:00
# -----------------------------
EMBEDDED_LATEST_DT = pd.Timestamp("2025-11-27 07:00:00")
EMBEDDED_DEFAULTS = {'AQI': 137.625,
 'AQI_lag1': 139.875,
 'AQI_lag12': 72.875,
 'AQI_lag2': 124.0,
 'AQI_lag3': 121.75,
 'AQI_lag6': 124.0,
 'BLH': 749.6999999999999,
 'BLH_lag1': 720.4250000000001,
 'CO': 0.9625,
 'CO_24h': 0.85,
 'CO_24h_lag1': 0.85,
 'CO_lag1': 0.9,
 'CloudCover': 0.0025,
 'CloudCover_lag1': 0.0,
 'Humidity': 49.175,
 'Humidity_lag1': 50.824999999999996,
 'NO2': 41.25,
 'NO2_24h': 49.0,
 'NO2_24h_lag1': 49.5,
 'NO2_lag1': 42.875,
 'O3': 5.166666666666667,
 'O3_24h': 10.73392857142857,
 'O3_24h_lag1': 10.73392857142857,
 'O3_8h': 53.0,
 'O3_8h_24h': 43.875,
 'O3_8h_24h_lag1': 43.875,
 'O3_8h_lag1': 53.0,
 'O3_lag1': 6.333333333333333,
 'PM10': 130.75,
 'PM10_24h': 117.25,
 'PM10_24h_lag1': 116.375,
 'PM10_lag1': 126.75,
 'PM2.5': 104.375,
 'PM2.5_24h': 79.25,
 'PM2.5_24h_lag1': 78.375,
 'PM2.5_lag1': 106.0,
 'Pressure': 961.4499999999999,
 'Pressure_lag1': 962.8875,
 'SO2': 6.875,
 'SO2_24h': 4.375,
 'SO2_24h_lag1': 4.125,
 'SO2_lag1': 5.375,
 'Temperature': 17.09625,
 'Temperature_lag1': 16.4225,
 'WindDir': 129.25,
 'WindDir_lag1': 95.75,
 'WindSpeed': 2.10875,
 'WindSpeed_lag1': 2.0812500000000003,
 'dayofweek': 3.0,
 'hour': 7.0,
 'is_weekend': 0.0,
 'latitude': 30.6463375,
 'longitude': 104.0793,
 'month': 11.0}


# -----------------------------
# Inference
# -----------------------------
def predict_with_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    feature_cols = bundle["feature_cols"]
    X_use = X[feature_cols]
    base_names = bundle["base_model_names"]
    stack = np.vstack([bundle["base_models"][m].predict(X_use) for m in base_names]).T
    return bundle["meta_model"].predict(stack)


# -----------------------------
# Session state init
# -----------------------------
if "defaults" not in st.session_state:
    st.session_state["defaults"] = EMBEDDED_DEFAULTS.copy()

if "latest_dt" not in st.session_state:
    st.session_state["latest_dt"] = EMBEDDED_LATEST_DT

if "last_pred_df" not in st.session_state:
    st.session_state["last_pred_df"] = None


# UI navigation state (not bound to any widget key)
if "view" not in st.session_state:
    st.session_state["view"] = "Input"

# Manual datetime (used to refresh time features); any date is allowed
if "manual_dt" not in st.session_state:
    # Default is aligned with built-in defaults timestamp (stable & reproducible)
    st.session_state["manual_dt"] = EMBEDDED_LATEST_DT

# -----------------------------
# Header
# -----------------------------
st.title("Chengdu Air Quality Multi-horizon Prediction (1h, 6h, 12h)")
st.markdown(
    '<span class="pill">Manual Input</span>'
    '<span class="pill">Built-in defaults</span>'
    '<span class="pill">Datetime → Time Features</span>'
    '<span class="pill">1/6/12h</span>',
    unsafe_allow_html=True
)


# -----------------------------
# Load models
# -----------------------------
try:
    bundles = load_all_bundles()
except Exception as e:
    st.error("❌ Failed to load models. Please check the models/ folder (model_ensemble_h1/h6/h12.joblib).")
    st.exception(e)
    st.stop()

feature_cols = bundles[1]["feature_cols"]

# -----------------------------
# Force defaults to Built-in defaults on first load (avoid old Session State carrying over)
# -----------------------------
APP_VERSION = "built_in_defaults_v6"
if st.session_state.get("_app_version") != APP_VERSION:
    st.session_state["_app_version"] = APP_VERSION
    st.session_state["defaults"] = EMBEDDED_DEFAULTS.copy()
    st.session_state["latest_dt"] = EMBEDDED_LATEST_DT
    st.session_state["manual_dt"] = EMBEDDED_LATEST_DT
    st.session_state["last_pred_df"] = None

    # Overwrite all feature widgets to built-in defaults (first load only)
    for feat in feature_cols:
        st.session_state[f"feat__{feat}"] = float(st.session_state["defaults"].get(feat, 0.0))



# -----------------------------
# Initialize inputs
# -----------------------------
# 1) Initialize all features with built-in defaults
for feat in feature_cols:
    k = f"feat__{feat}"
    if k not in st.session_state:
        if feat in st.session_state["defaults"]:
            st.session_state[k] = float(st.session_state["defaults"][feat])
        else:
            st.session_state[k] = 0.0

# 2) Override the 4 time features with manual_dt so locked values match datetime_input
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
    st.success("Models loaded ✅")

    horizons = st.multiselect("⏱️ Forecast horizons", [1, 6, 12], default=[1, 6, 12])

    st.divider()
    st.subheader("🗓️ datetime_input (select date & time)")

    # Allow selecting any date and time (to refresh the 4 time features)
    md0 = st.session_state["manual_dt"]
    init_date = md0.date()
    init_hour = int(md0.hour)

    pick_date = st.date_input("Date", value=init_date)

    # Hour-only selection (whole hours only)
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
        st.session_state["defaults"] = EMBEDDED_DEFAULTS.copy()
        st.session_state["latest_dt"] = EMBEDDED_LATEST_DT

        # One-click reset: overwrite all inputs with built-in defaults
        for feat in feature_cols:
            kk = f"feat__{feat}"
            if feat in st.session_state["defaults"]:
                st.session_state[kk] = float(st.session_state["defaults"][feat])

        # After reset, still refresh time features using manual_dt (keep locking logic consistent)
        _apply_time_from_manual_dt()

        st.toast("Defaults applied ✅", icon="✅")


    st.divider()
    st.subheader("ℹ️ Info")
    st.write(f"Built-in defaults timestamp: **{st.session_state['latest_dt']}**")
    st.write(f"Manual datetime: **{st.session_state['manual_dt']}**")
    st.write(f"Features required: **{len(feature_cols)}**")


# -----------------------------
# Feature grouping (with icons)
# -----------------------------
def group_features(cols: list[str]) -> dict[str, list[str]]:
    groups = {
        "🧭 Core (AQI)": [],
        "🕒 Time (locked)": [],
        "📉 AQI lags": [],
        "🧪 Pollutants (current)": [],
        "🧪 Pollutants (lag1)": [],
        "🌤️ Meteorology (current)": [],
        "🌤️ Meteorology (lag1)": [],
        "📦 Others": [],
    }

    pollutants_current = {
        "CO", "NO2", "O3", "O3_8h", "PM10", "PM2.5", "SO2",
        "CO_24h", "NO2_24h", "O3_24h", "O3_8h_24h", "PM10_24h", "PM2.5_24h", "SO2_24h"
    }
    meteo_base = {"Temperature", "Humidity", "WindSpeed", "WindDir", "Pressure", "BLH", "CloudCover"}

    for c in cols:
        if c == "AQI":
            groups["🧭 Core (AQI)"].append(c)
        elif c in {"longitude", "latitude"}:
            # Keep in feature vector (built-in defaults) but hide from UI.
            continue
        elif c in {"hour", "dayofweek", "month", "is_weekend"}:
            groups["🕒 Time (locked)"].append(c)
        elif c.startswith("AQI_lag"):
            groups["📉 AQI lags"].append(c)
        elif c in pollutants_current:
            groups["🧪 Pollutants (current)"].append(c)
        elif c.endswith("_lag1") and c.replace("_lag1", "") in pollutants_current:
            groups["🧪 Pollutants (lag1)"].append(c)
        elif c in meteo_base:
            groups["🌤️ Meteorology (current)"].append(c)
        elif c.endswith("_lag1") and c.replace("_lag1", "") in meteo_base:
            groups["🌤️ Meteorology (lag1)"].append(c)
        else:
            groups["📦 Others"].append(c)

    return {k: v for k, v in groups.items() if v}


groups = group_features(feature_cols)


# -----------------------------
# Widgets (Time features locked)
# -----------------------------
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

def render_widget(container, feat: str):
    key = f"feat__{feat}"

    # Lock the 4 time features: can only be refreshed via sidebar datetime_input
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
            y = float(predict_with_bundle(bundles[h], X)[0])
            outputs.append({"horizon_hours": h, "predicted_AQI": y})

        pred_df = pd.DataFrame(outputs).sort_values("horizon_hours").reset_index(drop=True)
        st.session_state["last_pred_df"] = pred_df

        # Auto-jump to Results page
        st.session_state["view"] = "Results"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

else:
    st.subheader("📈 Prediction Results")

    # Simple navigation back to inputs (no radio/selector widget)
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
            # Common AQI category names (0-500+)
            if aqi_value <= 50:
                return "Good"
            if aqi_value <= 100:
                return "Moderate"
            return "Unhealthy"

        # Baseline AQI for delta display (if available in inputs)
        baseline_aqi = None
        try:
            if "feat__AQI" in st.session_state:
                baseline_aqi = float(st.session_state["feat__AQI"])
        except Exception:
            baseline_aqi = None

        m1, m2, m3 = st.columns(3)
        for col, h in zip([m1, m2, m3], [1, 6, 12]):
            if (pred_df["horizon_hours"] == h).any():
                v = float(pred_df.loc[pred_df["horizon_hours"] == h, "predicted_AQI"].iloc[0])

                # Show small delta + arrow next to the main value (Streamlit metric delta)
                if baseline_aqi is None or (isinstance(baseline_aqi, float) and np.isnan(baseline_aqi)):
                    col.metric(f"t+{h}h AQI", f"{v:.2f}")
                else:
                    delta = v - baseline_aqi
                    col.metric(
                        f"t+{h}h AQI",
                        f"{v:.2f}",
                        delta=f"{delta:+.2f}",
                        delta_color="normal",
                    )

                # Make level more prominent
                col.markdown(
                    f"<div style='font-size:1.05rem; font-weight:650; margin-top:0.25rem;'>"
                    f"Level: {aqi_category(v)}</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.write("**Forecast curve**")
        st.line_chart(pred_df.set_index("horizon_hours"))
