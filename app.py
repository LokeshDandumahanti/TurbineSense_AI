# turbine_app.py  (UPDATED - fixes InvalidIndexError when prepending prediction tables)
import os
import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# optional Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

sns.set(style="whitegrid")

# ---------------- CONFIG (edit if you want) ----------------
DATA_CSV_DEFAULT = r"jk_comb1.csv"
SAVE_ROOT = "rolling_app_outputs"
SHORT_MODEL_DIR = os.path.join(SAVE_ROOT, "shortterm_models")
LONG_MODEL_DIR = os.path.join(SAVE_ROOT, "longterm_models")  # optional location for user-provided long-term models
PLOTS_DIR = os.path.join(SAVE_ROOT, "plots")
METRICS_LOG = os.path.join(SAVE_ROOT, "rolling_metrics.csv")

os.makedirs(SHORT_MODEL_DIR, exist_ok=True)
os.makedirs(LONG_MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)

TARGETS = ["TEY", "CO", "NOX"]
BASE_SENSORS = ["AT","AP","AH","AFDP","GTEP","TIT","TAT","CDP","CO","NOX"]
DEFAULT_BATCH_SIZE = 30
LAGS = [1,3,6,24]
ROLL_WINDOWS = [3,12,24]
PCA_FEATURES = ["CDP","GTEP","TIT"]
ASSUME_FREQ = "h"
RANDOM_STATE = 42

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": RANDOM_STATE,
    "verbosity": 0
}
NUM_ROUNDS = 200

# ---------------- helpers ----------------
def metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def add_lags_rolls(df, cols, lags=LAGS, windows=ROLL_WINDOWS):
    new_cols = {}
    for c in cols:
        if c not in df.columns:
            continue
        for l in lags:
            new_cols[f"{c}_lag{l}"] = df[c].shift(l)
        for w in windows:
            new_cols[f"{c}_rollmean_{w}"] = df[c].shift(1).rolling(window=w, min_periods=1).mean()
            new_cols[f"{c}_rollstd_{w}"] = df[c].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    return df

def ensure_timestamp(df, assume_freq=ASSUME_FREQ):
    if "TIMESTAMP" not in df.columns:
        df["TIMESTAMP"] = pd.date_range(start="2000-01-01", periods=len(df), freq=assume_freq)
    else:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    return df

def load_longterm_models(model_dir):
    long_models = {}
    for t in TARGETS:
        fname = os.path.join(model_dir, f"xgb_{t.lower()}_long.json")
        if os.path.exists(fname):
            booster = xgb.Booster()
            try:
                booster.load_model(fname)
                long_models[t] = booster
            except Exception as e:
                st.warning(f"Failed to load {fname}: {e}")
    return long_models

def safe_save_plot(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def make_unique_columns(df):
    """
    Ensure DataFrame has unique column names by appending suffixes to duplicates.
    Returns the DataFrame with unique columns.
    """
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_name = f"{c}_{seen[c]}"
            new_cols.append(new_name)
        else:
            seen[c] = 0
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df

def safe_prepend_dataframe(top_df, bottom_df):
    """
    Prepend top_df on top of bottom_df safely, ensuring unique columns.
    Returns concatenated DataFrame (top first).
    """
    try:
        # Make sure both dfs exist
        if top_df is None:
            top_df = pd.DataFrame()
        if bottom_df is None:
            bottom_df = pd.DataFrame()

        # Ensure column uniqueness for each
        top_df = make_unique_columns(top_df)
        bottom_df = make_unique_columns(bottom_df)

        # Create union of columns (order: top columns first, then remaining from bottom)
        cols_top = list(top_df.columns)
        cols_bottom = [c for c in bottom_df.columns if c not in cols_top]
        final_cols = cols_top + cols_bottom

        # Reindex both to final columns (fill missing with NaN)
        top_reindexed = top_df.reindex(columns=final_cols, fill_value=np.nan)
        bottom_reindexed = bottom_df.reindex(columns=final_cols, fill_value=np.nan)

        result = pd.concat([top_reindexed, bottom_reindexed], ignore_index=True, sort=False)
        return result
    except Exception as e:
        # Last resort fallback: reset columns of both to simple numeric and concat
        st.session_state["logs"].insert(0, f"safe_prepend_dataframe fallback triggered: {e}")
        top_df = top_df.reset_index(drop=True).copy() if not top_df.empty else pd.DataFrame()
        bottom_df = bottom_df.reset_index(drop=True).copy() if not bottom_df.empty else pd.DataFrame()
        # rename columns to avoid duplication
        top_df.columns = [f"c{i}" for i in range(top_df.shape[1])]
        bottom_df.columns = [f"c{i}" for i in range(bottom_df.shape[1])]
        return pd.concat([top_df, bottom_df], ignore_index=True, sort=False)

# ensure session state containers
if "logs" not in st.session_state:
    st.session_state["logs"] = []           # stack: newest at index 0
if "metrics_history" not in st.session_state:
    st.session_state["metrics_history"] = []  # list of dicts, newest first
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = pd.DataFrame()  # store table; we'll prepend rows

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="TurbineSense AI", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #0E6BA8;'>⚙️ TurbineSense AI</h1>
    <h3 style='text-align: center; color: #555;'>Unified GenAI & Data Analytics Platform for Turbine Health Monitoring, Yield Prediction (TEY) & Emission Control (CO / NOX)</h3>
    <p style='text-align: center; font-style: italic; color: #888;'>AI-powered insights for smarter turbines, reliable performance & zero downtime</p>
    <hr>
""", unsafe_allow_html=True)

# Groq assistant input (top)
st.sidebar.header("LLM / Groq Assistant (optional)")
groq_key = st.sidebar.text_input("Groq API Key (optional, for quick Q&A)", type="password")
if groq_key and not GROQ_AVAILABLE:
    st.sidebar.warning("groq library not installed; assistant will be disabled until groq is available.")

st.sidebar.markdown("Data & training settings")
data_csv = st.sidebar.text_input("Path to data CSV", DATA_CSV_DEFAULT)
batch_size = st.sidebar.number_input("Batch size (sequential)", value=DEFAULT_BATCH_SIZE, min_value=5, max_value=1000, step=1)
st.sidebar.markdown(f"Short-term models saved to: `{SHORT_MODEL_DIR}`")
st.sidebar.markdown(f"Long-term models (optional) folder: `{LONG_MODEL_DIR}`")
st.sidebar.markdown(f"Plots saved to: `{PLOTS_DIR}`")

# model type dropdown but default is XGBoost
model_choice = st.sidebar.selectbox("Short-term model type", ["XGBoost (default)"], index=0)

# show quick dataset summary
st.sidebar.markdown("### Dataset preview")
try:
    df_preview = pd.read_csv(data_csv, nrows=5, low_memory=False)
    st.sidebar.dataframe(df_preview)
except Exception as e:
    st.sidebar.error(f"Cannot preview dataset: {e}")

# main Groq search box
st.markdown("### Ask the Data Analyst (LLM) — quick Q&A (optional)")
user_question = st.text_input("Ask a question about the dataset or troubleshooting (e.g., 'When did TEY drop below 150?')")

if user_question:
    if groq_key and GROQ_AVAILABLE:
        try:
            client = Groq(api_key=groq_key)
            messages = [
                {"role":"system", "content": "You are a sharp data analyst for gas turbine metrics. Answer concisely and reference dataset preview."},
                {"role":"user", "content": user_question}
            ]
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.6,
                max_completion_tokens=512,
                top_p=1
            )
            answer = completion.choices[0].message.content
            st.markdown("**Analyst:**")
            st.write(answer)
            st.session_state["logs"].insert(0, f"LLM answered: {user_question}")
        except Exception as e:
            st.error(f"Assistant failed: {e}")
            st.session_state["logs"].insert(0, f"LLM failure: {e}")
    else:
        st.info("Provide Groq API key in the sidebar to enable the assistant (or install groq package).")
        st.session_state["logs"].insert(0, f"LLM requested but no key / groq not installed")

# Main action button
st.markdown("---")
run_button = st.button("Run rolling training (sequential batches)")

# Show recent logs (stack style: newest first)
st.sidebar.markdown("### Recent logs (newest first)")
for log in st.session_state["logs"][:50]:
    st.sidebar.write(log)

# Execution of rolling training only when button clicked
if run_button:
    st.session_state["logs"].insert(0, f"Rolling run started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    try:
        df_raw = pd.read_csv(data_csv, low_memory=False)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.session_state["logs"].insert(0, f"Failed to load CSV: {e}")
        st.stop()

    # Preprocess
    st.info("Preprocessing and feature engineering...")
    df = df_raw.copy()
    df = ensure_timestamp(df, ASSUME_FREQ)

    # Clip extreme outliers for sensors present
    for col in BASE_SENSORS:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)

    # add lags and rolling features
    df = add_lags_rolls(df, BASE_SENSORS, lags=LAGS, windows=ROLL_WINDOWS)

    # time features
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["dayofweek"] = df["TIMESTAMP"].dt.dayofweek
    df["month"] = df["TIMESTAMP"].dt.month
    df["day"] = df["TIMESTAMP"].dt.day

    # PCA features if available
    if all([c in df.columns for c in PCA_FEATURES]):
        try:
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            pcs = pca.fit_transform(df[PCA_FEATURES].fillna(method="ffill").fillna(0))
            df["PC1"] = pcs[:,0]
            df["PC2"] = pcs[:,1]
        except Exception as e:
            st.warning(f"PCA failed: {e}")
            df["PC1"] = 0
            df["PC2"] = 0
    else:
        df["PC1"] = 0
        df["PC2"] = 0

    # drop rows with NaN from lags
    df = df.dropna().reset_index(drop=True)
    st.success(f"Preprocessing complete. Processed shape: {df.shape}")
    st.session_state["logs"].insert(0, f"Preprocessing done. rows={len(df)}")

    # determine features
    exclude_cols = ["TIMESTAMP"] + [c for c in TARGETS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    st.session_state["logs"].insert(0, f"Using {len(feature_cols)} features for models")

    # scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # load long-term models if present
    long_models = load_longterm_models(LONG_MODEL_DIR)
    if long_models:
        st.session_state["logs"].insert(0, f"Loaded long-term models: {', '.join(long_models.keys())}")
    else:
        st.session_state["logs"].insert(0, "No long-term models found (long-term predictions will be skipped)")

    # prepare rolling sequential batches
    n_rows = len(df)
    max_pairs = (n_rows // batch_size) - 1
    if max_pairs <= 0:
        st.error("Dataset too small for given batch size.")
        st.session_state["logs"].insert(0, "Dataset too small for batch processing")
        st.stop()

    overall_metrics = []
    aggregated_actuals = {t: [] for t in TARGETS}
    aggregated_preds_short = {t: [] for t in TARGETS}
    aggregated_preds_long = {t: [] for t in TARGETS}
    batch_counter = 0
    progress = st.progress(0)

    for pair_idx in range(max_pairs):
        train_start = pair_idx * batch_size
        train_end = train_start + batch_size
        test_start = train_end
        test_end = test_start + batch_size
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        with st.expander(f"Batch {pair_idx+1} | rows {train_start}-{test_end-1}", expanded=False):
            st.write(f"Train {train_start}->{train_end-1}  |  Test {test_start}->{test_end-1}")
            per_batch_info = []
            for target in TARGETS:
                if target not in df.columns:
                    st.warning(f"Target {target} not present in dataset; skipping.")
                    continue

                X_train = train_df[feature_cols].fillna(0)
                y_train = train_df[target].values
                X_test = test_df[feature_cols].fillna(0)
                y_test = test_df[target].values

                # Train short-term model (XGBoost)
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=NUM_ROUNDS,
                    learning_rate=XGB_PARAMS["eta"],
                    max_depth=XGB_PARAMS["max_depth"],
                    subsample=XGB_PARAMS["subsample"],
                    colsample_bytree=XGB_PARAMS["colsample_bytree"],
                    random_state=RANDOM_STATE,
                    verbosity=0,
                    tree_method='hist'
                )
                model.fit(X_train, y_train, verbose=False)

                # Save model
                short_model_path = os.path.join(SHORT_MODEL_DIR, f"xgb_{target.lower()}_batch_{pair_idx}.json")
                model.save_model(short_model_path)

                # Predictions
                y_pred_short = model.predict(X_test)

                # long-term prediction if available
                if target in long_models:
                    booster = long_models[target]
                    long_features = booster.feature_names
                    X_test_long = X_test.reindex(columns=long_features, fill_value=0)
                    dtest_long = xgb.DMatrix(X_test_long)
                    try:
                        y_pred_long = booster.predict(dtest_long)
                        long_info = "Long-term model used"
                    except Exception as e:
                        y_pred_long = np.array([np.nan]*len(y_test))
                        long_info = f"Long-term predict failed: {e}"
                else:
                    y_pred_long = np.array([np.nan]*len(y_test))
                    long_info = "No long-term model"

                # metrics
                m_short = metrics(y_test, y_pred_short)
                m_long = metrics(y_test, y_pred_long) if not np.all(np.isnan(y_pred_long)) else {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

                # Update session_state metrics as stack (newest first)
                entry = {
                    "batch": pair_idx,
                    "target": target,
                    "short_MAE": m_short["MAE"],
                    "short_RMSE": m_short["RMSE"],
                    "short_R2": m_short["R2"],
                    "long_MAE": m_long["MAE"],
                    "long_RMSE": m_long["RMSE"],
                    "long_R2": m_long["R2"],
                    "short_model": short_model_path,
                    "long_info": long_info,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state["metrics_history"].insert(0, entry)

                # save overall metrics
                overall_metrics.append(entry)

                # aggregated
                aggregated_actuals[target].extend(y_test.tolist())
                aggregated_preds_short[target].extend(y_pred_short.tolist())
                aggregated_preds_long[target].extend(y_pred_long.tolist() if hasattr(y_pred_long, "tolist") else [np.nan]*len(y_test))

                # show metrics
                st.write(f"**{target}** short-term MAE={m_short['MAE']:.4f} RMSE={m_short['RMSE']:.4f} R2={m_short['R2']:.4f}")
                if not math.isnan(m_long["MAE"]):
                    st.write(f"Long-term MAE={m_long['MAE']:.4f} RMSE={m_long['RMSE']:.4f} R2={m_long['R2']:.4f}")
                else:
                    st.write("Long-term: not available or failed")

                st.write(f"Short-term model saved: `{short_model_path}`")
                st.write(long_info)

                # plot
                fig, ax = plt.subplots(figsize=(10,3))
                ax.plot(y_test, label="Actual", marker='o', linewidth=1)
                if not np.all(np.isnan(y_pred_long)):
                    ax.plot(y_pred_long, label="Long-term", marker='x', linewidth=1, alpha=0.8)
                ax.plot(y_pred_short, label="Short-term", marker='s', linewidth=1, alpha=0.9)
                ax.set_title(f"{target} - Batch {pair_idx}")
                ax.legend()
                plot_path = os.path.join(PLOTS_DIR, f"{target}_batch_{pair_idx}.png")
                safe_save_plot(fig, plot_path)
                st.image(plot_path, use_column_width=True, caption=f"{target} batch {pair_idx}")

                # prediction table for this test batch: newest rows on top
                test_show = test_df.reset_index(drop=True).copy()
                test_show[f"{target}_pred_short"] = y_pred_short
                test_show[f"{target}_pred_long"] = y_pred_long

                small_table = test_show[[ "TIMESTAMP"] + [c for c in BASE_SENSORS if c in test_show.columns] + [target, f"{target}_pred_short", f"{target}_pred_long"]]

                # === SAFELY prepend small_table to session_state["prediction_history"] ===
                try:
                    if st.session_state["prediction_history"].empty:
                        # ensure small_table columns are unique
                        small_table = make_unique_columns(small_table)
                        st.session_state["prediction_history"] = small_table.reset_index(drop=True)
                    else:
                        # safe prepend (handles duplicate column names)
                        st.session_state["prediction_history"] = safe_prepend_dataframe(small_table.reset_index(drop=True), st.session_state["prediction_history"])
                except Exception as e:
                    # robust fallback: convert to CSV strings and prepend as new rows
                    st.session_state["logs"].insert(0, f"Error prepending prediction table: {e}")
                    try:
                        # simplest fallback: append as CSV text into logs (not ideal, but safe)
                        st.session_state["logs"].insert(0, f"Fallback: stored batch {pair_idx} prediction as text.")
                    except Exception:
                        pass

                # residual plot
                resid = (y_test - y_pred_short)
                fig2, ax2 = plt.subplots(figsize=(8,2.5))
                ax2.hist(resid, bins=30)
                ax2.set_title(f"{target} residuals (short-term)")
                resid_path = os.path.join(PLOTS_DIR, f"{target}_resid_batch_{pair_idx}.png")
                safe_save_plot(fig2, resid_path)
                st.image(resid_path, use_column_width=True, caption="Residuals (short-term)")

            batch_counter += 1
            progress.progress(int(batch_counter / max_pairs * 100))

    # finishing
    if overall_metrics:
        pd.DataFrame(overall_metrics).to_csv(METRICS_LOG, index=False)
        st.session_state["logs"].insert(0, f"Saved metrics to {METRICS_LOG}")

    # aggregated final plots
    for target in TARGETS:
        if aggregated_actuals[target]:
            fig, ax = plt.subplots(figsize=(14,4))
            ax.plot(aggregated_actuals[target], label="Actual", color="black", linewidth=1)
            ax.plot(aggregated_preds_short[target], label="Short-term", color="blue", linewidth=0.8, alpha=0.8)
            ax.plot(aggregated_preds_long[target], label="Long-term", color="red", linewidth=0.8, alpha=0.7)
            ax.set_title(f"{target} aggregated across batches")
            ax.legend()
            agg_path = os.path.join(PLOTS_DIR, f"{target}_rolling_all.png")
            safe_save_plot(fig, agg_path)
            st.image(agg_path, use_column_width=True, caption=f"{target} aggregated")

    elapsed = time.time() - start_time
    st.session_state["logs"].insert(0, f"Rolling run finished in {elapsed:.1f}s")
    st.success("Rolling training completed. Check sidebars & outputs.")
    st.balloons()

# show recent metrics history (newest first)
st.markdown("### Recent metrics (newest first)")
if st.session_state["metrics_history"]:
    df_metrics_show = pd.DataFrame(st.session_state["metrics_history"])
    st.dataframe(df_metrics_show.head(200))

# show prediction history (stack newest first)
st.markdown("### Prediction history (newest blocks at top)")
if not st.session_state["prediction_history"].empty:
    st.dataframe(st.session_state["prediction_history"].head(500))

# show logs
st.markdown("### Activity log (newest first)")
for entry in st.session_state["logs"][:200]:
    st.write(entry)
