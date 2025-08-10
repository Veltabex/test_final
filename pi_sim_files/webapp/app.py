# app.py — IoT Malware Detector (SVM) with Baseline + Drift + Folder Prediction
# Paths set for Raspberry Pi (32‑bit Raspberry Pi Desktop in VirtualBox)

import os, re, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, redirect, url_for, render_template, flash

import matplotlib
matplotlib.use("Agg")               # headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ─────────────── PATHS (your layout) ───────────────
BASE_DIR      = "/home/pi/Desktop/webapp"
DATA_DIR      = os.path.join(BASE_DIR, "test_data")      # where your CSVs live
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")      # selected_features.json, label_map.json, (optional) medians.json
MODELS_DIR    = os.path.join(BASE_DIR, "models")         # svm_model.pkl, scaler.pkl (or iot_malware_svm_subset_*.pkl)
STATIC_DIR    = os.path.join(BASE_DIR, "static")         # CSS, images
PLOTS_DIR     = os.path.join(STATIC_DIR, "plots")        # where we save plots
BASELINE_DIR  = os.path.join(BASE_DIR, "baseline")       # baseline_mean_std.json

# Artifacts
SELECTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "selected_features.json")
LABEL_MAP_PATH         = os.path.join(ARTIFACTS_DIR, "label_map.json")
MEDIANS_PATH           = os.path.join(ARTIFACTS_DIR, "medians.json")       # optional (training medians for imputation)

# Model/scaler (first existing path wins)
SCALER_CANDIDATES = [
    os.path.join(MODELS_DIR, "iot_malware_svm_subset_scaler.pkl"),
    os.path.join(MODELS_DIR, "svm_scaler.pkl"),
    os.path.join(MODELS_DIR, "scaler.pkl"),
]
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "iot_malware_svm_subset_model.pkl"),
    os.path.join(MODELS_DIR, "svm_model.pkl"),
    os.path.join(MODELS_DIR, "model.pkl"),
]

# Options
SUBSET_N       = 5000           # cap rows per run; set None to use all
RANDOM_STATE   = 42
ALLOWED_EXT    = {".csv"}
MAX_FILE_SIZE  = 200 * 1024 * 1024
ENABLE_NGROK   = os.environ.get("ENABLE_NGROK", "0") == "1"
NGROK_TOKEN    = os.environ.get("NGROK_AUTHTOKEN", "")

# FS
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)

# Flask
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "change-this-secret"

# ─────────────── Load core artifacts ───────────────
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these exist:\n" + "\n".join(paths))

SELECTED_FEATURES = _load_json(SELECTED_FEATURES_PATH)
LABEL_MAP         = _load_json(LABEL_MAP_PATH)

TRAIN_MEDIANS = None
if os.path.exists(MEDIANS_PATH):
    TRAIN_MEDIANS = pd.Series(_load_json(MEDIANS_PATH))
else:
    warnings.warn("medians.json not found; falling back to per‑file medians.")

SCALER_PATH = _first_existing(SCALER_CANDIDATES)
MODEL_PATH  = _first_existing(MODEL_CANDIDATES)
scaler = joblib.load(SCALER_PATH)
model  = joblib.load(MODEL_PATH)

if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(SELECTED_FEATURES):
    raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but selected_features.json has {len(SELECTED_FEATURES)}.")

BASELINE_PATH = os.path.join(BASELINE_DIR, "baseline_mean_std.json")   # stores scaled mean & std

# ─────────────── Helpers ───────────────
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def safe_save_upload(file_storage, dest_dir=DATA_DIR):
    fname = file_storage.filename
    if not allowed_file(fname):
        raise ValueError("Only .csv files are allowed.")
    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"{fname} is too large ({size} bytes). Limit: {MAX_FILE_SIZE} bytes.")
    path = os.path.join(dest_dir, fname)
    file_storage.save(path)
    return path

def list_csvs(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".csv")])

def load_csv_folder(folder: str) -> pd.DataFrame:
    frames = []
    for f in list_csvs(folder):
        p = os.path.join(folder, f)
        df = pd.read_csv(p, low_memory=False)
        df["__source_file__"] = f
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSVs found in {folder}")
    return pd.concat(frames, ignore_index=True)

def infer_labels(df: pd.DataFrame, fname: str):
    # prefer explicit column if present
    for col in ["label", "Label", "y", "target"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("Int64"), "from_column"
    # filename hint
    m = re.search(r"(Mirai|BASHLITE|benign)", os.path.basename(fname), re.IGNORECASE)
    if m:
        key = m.group(1)
        key_std = key.capitalize() if key.lower() != "benign" else "benign"
        if key_std in LABEL_MAP:
            return pd.Series(np.full(len(df), LABEL_MAP[key_std], dtype=int)), "from_filename"
    return None, "none"

def clean_and_align(df: pd.DataFrame, file_hint: str) -> pd.DataFrame:
    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        # be strict to avoid garbage scaling
        raise ValueError(f"{file_hint}: missing features: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = df[SELECTED_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # fill NaNs using training medians when available (more faithful)
    if TRAIN_MEDIANS is not None:
        train_meds = TRAIN_MEDIANS.reindex(SELECTED_FEATURES)
        fallback = X.median(numeric_only=True)
        X = X.fillna(train_meds.fillna(fallback))
    else:
        X = X.fillna(X.median(numeric_only=True))

    X = X.astype(np.float64)

    if SUBSET_N is not None and len(X) > SUBSET_N:
        X = X.sample(SUBSET_N, random_state=RANDOM_STATE)
    return X

# Baseline: store mean & std in **scaled space** so drift uses the same scale as the model
def save_baseline_scaled(X_scaled: np.ndarray):
    mean_vec = np.mean(X_scaled, axis=0).tolist()
    std_vec  = np.std(X_scaled, axis=0).tolist()
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features": SELECTED_FEATURES,
        "scaled_mean": mean_vec,
        "scaled_std": std_vec
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_baseline_scaled():
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_drift(X_scaled: np.ndarray, baseline: dict) -> float:
    means = np.array(baseline["scaled_mean"], dtype=np.float64)
    stds  = np.array(baseline["scaled_std"], dtype=np.float64)
    stds[stds == 0] = 1e-9
    z = (X_scaled - means) / stds
    return float(np.nanmean(np.abs(z)))  # mean |z|

# Plots
def plot_preds_series(y_pred, out_path):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(len(y_pred)), y_pred, lw=1)
    ax.set_title("Predictions over Samples")
    ax.set_xlabel("Sample index"); ax.set_ylabel("Predicted label (0/1)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

def plot_pred_bar(y_pred, out_path):
    vals, cnts = np.unique(y_pred, return_counts=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(vals, cnts)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign (0)", "Malware (1)"])
    ax.set_title("Prediction Counts")
    for i, c in enumerate(cnts):
        ax.text(vals[i], c, str(c), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

def plot_cm(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Benign","Malware"]).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

# ─────────────── Routes ───────────────
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        files=list_csvs(DATA_DIR),
        n_features=len(SELECTED_FEATURES),
        subset=(SUBSET_N if SUBSET_N else "All"),
        baseline_loaded=os.path.exists(BASELINE_PATH)
    )

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part.")
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        flash("No selected file.")
        return redirect(url_for("index"))
    try:
        saved = safe_save_upload(f)
        flash(f"Uploaded: {os.path.basename(saved)}")
    except Exception as e:
        flash(f"Upload failed: {e}")
    return redirect(url_for("index"))

@app.route("/baseline", methods=["POST"])
def build_baseline():
    folder = request.form.get("baseline_folder", "").strip()
    if not folder:
        flash("Provide a benign folder path.")
        return redirect(url_for("index"))
    try:
        df = load_csv_folder(folder)
        X = clean_and_align(df, f"{folder}/*")
        Xs = scaler.transform(X)
        save_baseline_scaled(Xs)
        flash(f"Baseline built from {len(X)} rows in {folder}.")
    except Exception as e:
        flash(f"Baseline failed: {e}")
    return redirect(url_for("index"))

@app.route("/reset_baseline", methods=["GET"])
def reset_baseline():
    try:
        if os.path.exists(BASELINE_PATH):
            os.remove(BASELINE_PATH)
            flash("Baseline reset.")
        else:
            flash("No baseline file to remove.")
    except Exception as e:
        flash(f"Reset failed: {e}")
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    folder = request.form.get("predict_folder", "").strip()
    if not folder:
        flash("Provide a folder path containing CSVs.")
        return redirect(url_for("index"))

    results = {
        "samples": 0, "pred_time": None,
        "acc": None, "f1": None, "report": None,
        "drift": None
    }

    try:
        df = load_csv_folder(folder)
        # labels (best effort): if any file has a label column we’ll use it, else fall back to filename hint
        y_true_all = None
        if any(col in df.columns for col in ["label","Label","y","target"]):
            col = next(c for c in ["label","Label","y","target"] if c in df.columns)
            y_true_all = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        t0 = time.time()
        X = clean_and_align(df, f"{folder}/*")
        Xs = scaler.transform(X)
        y_pred = model.predict(Xs)
        pred_time = round(time.time() - t0, 4)

        # plots
        plot_preds_series(y_pred, os.path.join(PLOTS_DIR, "preds_series.png"))
        plot_pred_bar(y_pred,    os.path.join(PLOTS_DIR, "preds_bar.png"))

        # drift vs baseline (if exists)
        baseline = load_baseline_scaled()
        if baseline:
            results["drift"] = round(compute_drift(Xs, baseline), 4)

        # metrics if ground truth available
        if y_true_all is not None:
            y_true = y_true_all.loc[X.index].astype(int)
            rep = classification_report(y_true, y_pred, target_names=["Benign","Malware"], digits=4)
            cm = confusion_matrix(y_true, y_pred)
            acc = (cm.trace() / cm.sum()) if cm.sum() else 0.0
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
            plot_cm(y_true, y_pred, "Confusion Matrix", os.path.join(PLOTS_DIR, "confusion_matrix.png"))
            results.update({
                "samples": int(len(X)), "pred_time": pred_time,
                "acc": round(acc, 4), "f1": round(f1, 4), "report": rep
            })
        else:
            counts = pd.Series(y_pred).value_counts().sort_index().to_dict()
            rep_text = f"No ground‑truth labels found. Prediction counts: {counts}"
            results.update({"samples": int(len(X)), "pred_time": pred_time, "report": rep_text})

        flash(f"Prediction complete for folder: {folder}")
    except Exception as e:
        flash(f"Predict failed: {e}")
        results = None

    return render_template(
        "index.html",
        files=list_csvs(DATA_DIR),
        n_features=len(SELECTED_FEATURES),
        subset=(SUBSET_N if SUBSET_N else "All"),
        baseline_loaded=os.path.exists(BASELINE_PATH),
        results=results
    )

def maybe_start_ngrok():
    if not ENABLE_NGROK:
        return
    try:
        from pyngrok import ngrok, conf
        if NGROK_TOKEN:
            conf.get_default().auth_token = NGROK_TOKEN
        public_url = ngrok.connect(5000, "http").public_url
        print(f" * ngrok tunnel: {public_url}")
    except Exception as e:
        print(f"ngrok failed: {e}")

if __name__ == "__main__":
    print("Starting IoT Malware Detector…")
    maybe_start_ngrok()
    app.run(host="0.0.0.0", port=5000, debug=False)
