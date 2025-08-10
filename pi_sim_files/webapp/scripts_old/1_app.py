import os, json, uuid
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from joblib import load
from sklearn.metrics import classification_report
from pyngrok import ngrok

# ---------- CONFIG ----------
MODEL_DIR = "/home/pi/Desktop/iot/model"
MODEL_PATH = os.path.join(MODEL_DIR, "iot_malware_svm_subset_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "iot_malware_svm_subset_scaler.pkl")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(APP_DIR, "state")
BASELINE_PATH = os.path.join(STATE_DIR, "baseline_stats.json")
LAST_REPORT_PATH = os.path.join(STATE_DIR, "last_report.json")

SAMPLE_N = 5000  # sample up to 5k rows

os.makedirs(STATE_DIR, exist_ok=True)

# ---------- LOAD MODEL ----------
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# ---------- APP ----------
app = Flask(__name__)

def safe_sample(df: pd.DataFrame, n=SAMPLE_N):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)

def keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # drop non-numerics; keep only numeric features (your SVM expects same columns used in training)
    num = df.select_dtypes(include=[np.number]).copy()
    # drop label-ish columns if present
    for c in ["label", "Label", "y", "target"]:
        if c in num.columns:
            num = num.drop(columns=[c])
    return num

def compute_baseline_stats(X_scaled: np.ndarray):
    return {
        "mean": np.mean(X_scaled, axis=0).tolist(),
        "std":  np.std(X_scaled, axis=0).tolist()
    }

def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def drift_score(X_scaled: np.ndarray, baseline: dict) -> float:
    """Simple distance between current mean/std and saved baseline."""
    bmean = np.array(baseline["mean"])
    bstd  = np.array(baseline["std"]) + 1e-8
    cmean = np.mean(X_scaled, axis=0)
    cstd  = np.std(X_scaled, axis=0) + 1e-8

    mean_dist = np.linalg.norm((cmean - bmean) / bstd)
    std_dist  = np.linalg.norm((cstd - bstd) / bstd)
    return float(mean_dist + 0.5 * std_dist)

@app.route("/")
def index():
    # show current state
    has_baseline = os.path.exists(BASELINE_PATH)
    public_url = ngrok.get_tunnels()[0].public_url if ngrok.get_tunnels() else "http://localhost:5000"
    return render_template("index.html",
                           has_baseline=has_baseline,
                           public_url=public_url)

@app.route("/toggle", methods=["POST"])
def toggle():
    # purely visual toggle; frontend handles the UI
    return jsonify(ok=True)

@app.route("/baseline", methods=["POST"])
def baseline():
    if "file" not in request.files:
        return jsonify(ok=False, msg="No file uploaded"), 400
    f = request.files["file"]
    df = pd.read_csv(f)
    df = keep_numeric(df)
    df = safe_sample(df)
    if df.empty:
        return jsonify(ok=False, msg="No numeric data found"), 400

    X_scaled = scaler.transform(df.values)
    stats = compute_baseline_stats(X_scaled)
    save_json(BASELINE_PATH, stats)

    # Create a short series to draw a "signal" line (just some mean of batches)
    batch = min(50, len(X_scaled))
    seg = np.array_split(X_scaled[:batch], min(20, batch))
    signal = [float(np.mean(s)) for s in seg]

    return jsonify(ok=True, signal=signal, count=int(len(df)))

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify(ok=False, msg="No file uploaded"), 400

    df = pd.read_csv(request.files["file"])
    df_num = keep_numeric(df)
    df_num = safe_sample(df_num)
    if df_num.empty:
        return jsonify(ok=False, msg="No numeric data found"), 400

    X_scaled = scaler.transform(df_num.values)

    # Predictions
    y_pred = model.predict(X_scaled)
    # Try to infer labels if present for a quick report, otherwise fake zeros
    y_true = df["label"].values[:len(df_num)] if "label" in df.columns else np.zeros(len(df_num), dtype=int)

    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception:
        report = {}

    # Drift vs baseline if available
    base = load_baseline()
    dscore = drift_score(X_scaled, base) if base else None

    out = {
        "n_rows": int(len(df_num)),
        "pred_malware_rate": float(np.mean(y_pred)),
        "drift_score": dscore,
        "report": report
    }
    save_json(LAST_REPORT_PATH, out)
    return jsonify(ok=True, **out)

@app.route("/reset", methods=["POST"])
def reset():
    for p in [BASELINE_PATH, LAST_REPORT_PATH]:
        if os.path.exists(p):
            os.remove(p)
    return jsonify(ok=True)

def start_ngrok():
    if ngrok.get_tunnels():
        return
    # Use default authtoken configured via `ngrok config add-authtoken ...`
    url = ngrok.connect(5000).public_url
    print(f"[ngrok] Public URL: {url}")

if __name__ == "__main__":
    start_ngrok()
    app.run(host="0.0.0.0", port=5000, debug=False)
