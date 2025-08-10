# app.py — IoT Malware Prediction Web UI (SVM)
# Raspberry Pi paths kept the same; adds drift chip, bar plot, cache-busting

import os, re, json, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, redirect, url_for, render_template_string, flash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── PATHS ─────────────────────────────────────────────────────────────
BASE_DIR      = "/home/pi/Desktop/webapp"
DATA_DIR      = os.path.join(BASE_DIR, "test_data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
PLOTS_DIR     = os.path.join(STATIC_DIR, "plots")
BASELINE_DIR  = os.path.join(BASE_DIR, "baseline")

SELECTED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "selected_features.json")
LABEL_MAP_PATH         = os.path.join(ARTIFACTS_DIR, "label_map.json")
MEDIANS_PATH           = os.path.join(ARTIFACTS_DIR, "medians.json")  # optional

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

# ── OPTIONS ───────────────────────────────────────────────────────────
SUBSET_N      = 5000     # set None to use all rows
RANDOM_STATE  = 42
ALLOWED_EXT   = {".csv"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

ENABLE_NGROK   = False
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")

# ── SETUP ─────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-secret"

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these exist:\n" + "\n".join(paths))

# Load artifacts
SELECTED_FEATURES = _load_json(SELECTED_FEATURES_PATH)
LABEL_MAP = _load_json(LABEL_MAP_PATH)

TRAIN_MEDIANS = None
if os.path.exists(MEDIANS_PATH):
    TRAIN_MEDIANS = pd.Series(_load_json(MEDIANS_PATH))
else:
    warnings.warn("medians.json not found; using per-file medians.")

SCALER_PATH = _first_existing(SCALER_CANDIDATES)
MODEL_PATH  = _first_existing(MODEL_CANDIDATES)
scaler = joblib.load(SCALER_PATH)
model  = joblib.load(MODEL_PATH)

if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(SELECTED_FEATURES):
    raise ValueError(
        f"Scaler expects {scaler.n_features_in_} features but selected_features.json has {len(SELECTED_FEATURES)}."
    )

# ── HELPERS ───────────────────────────────────────────────────────────
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

def infer_labels(df: pd.DataFrame, fname: str):
    # Prefer explicit column
    for col in ["label", "Label", "y", "target"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("Int64"), "from_column"
    # fallback: filename tag
    m = re.search(r"(Mirai|BASHLITE|benign)", os.path.basename(fname), re.IGNORECASE)
    if m:
        key = m.group(1)
        key_std = key.capitalize() if key.lower() != "benign" else "benign"
        if key_std in LABEL_MAP:
            return pd.Series(np.full(len(df), LABEL_MAP[key_std], dtype=int)), "from_filename"
    return None, "none"

def clean_and_align(df: pd.DataFrame, file_hint: str):
    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{file_hint}: missing features: {missing[:10]}{'...' if len(missing)>10 else ''}")
    X = df[SELECTED_FEATURES].copy().replace([np.inf, -np.inf], np.nan)
    if TRAIN_MEDIANS is not None:
        train_meds = TRAIN_MEDIANS.reindex(SELECTED_FEATURES)
        fallback = X.median(numeric_only=True)
        fill_vals = train_meds.fillna(fallback)
        X = X.fillna(fill_vals)
    else:
        X = X.fillna(X.median(numeric_only=True))
    X = X.astype(np.float64)
    if SUBSET_N is not None and len(X) > SUBSET_N:
        X = X.sample(SUBSET_N, random_state=RANDOM_STATE)
    return X

# plots
def plot_cm(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","Malware"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

def plot_pred_counts(y_pred, out_path):
    counts = pd.Series(y_pred).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.index = counts.index.map({0:"Benign (0)",1:"Malware (1)"}).astype(str)
    ax.bar(counts.index, counts.values)
    ax.set_title("Prediction Counts"); ax.set_ylabel("Count")
    for i,v in enumerate(counts.values):
        ax.text(i, v + max(counts.values)*0.01, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

# baseline vec (scaled mean)
BASELINE_VEC_PATH = os.path.join(BASELINE_DIR, "baseline_mean.json")

def save_baseline_vector(Xs):
    mean_vec = np.mean(Xs, axis=0).tolist()
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features": SELECTED_FEATURES,
        "mean_vector": mean_vec
    }
    with open(BASELINE_VEC_PATH, "w") as f:
        json.dump(payload, f, indent=2)

def load_baseline_vector():
    if not os.path.exists(BASELINE_VEC_PATH):
        return None
    with open(BASELINE_VEC_PATH, "r") as f:
        return json.load(f)

def drift_value(Xs, baseline_payload):
    """Mean absolute difference between current scaled mean and baseline mean."""
    if not baseline_payload: 
        return None
    b = np.asarray(baseline_payload["mean_vector"], dtype=float)
    c = np.asarray(np.mean(Xs, axis=0), dtype=float)
    return float(np.mean(np.abs(c - b)))

# ── UI ────────────────────────────────────────────────────────────────
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IoT Malware Detection (SVM)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body{padding-top:20px}
    .badge-space{margin-right:6px}
    .plot-box{max-width:920px}
  </style>
</head>
<body class="bg-light">
<div class="container">
  <h3 class="mb-3">IoT Malware Detection — Web UI</h3>
  <p class="text-muted">Model: SVM | Features: {{ n_features }} | Subset: {{ subset }}</p>

  {% with messages = get_flashed_messages() %}{% if messages %}
    <div class="alert alert-info">{{ messages[-1] }}</div>{% endif %}{% endwith %}

  <div class="row">
    <div class="col-lg-6">
      <div class="card mb-3">
        <div class="card-header">Upload CSV</div>
        <div class="card-body">
          <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data" class="row g-3">
            <div class="col-sm-8"><input class="form-control" type="file" name="file" required></div>
            <div class="col-sm-4"><button class="btn btn-primary w-100" type="submit">Upload</button></div>
          </form>
          <small class="text-muted">Allowed: .csv | Stored in /test_data</small>
        </div>
      </div>

      <div class="card mb-3">
        <div class="card-header">1) Build/Update Baseline (Benign CSV)</div>
        <div class="card-body">
          <form method="post" action="{{ url_for('build_baseline') }}" class="row g-3">
            <div class="col-sm-8">
              <select class="form-select" name="filename">
                {% for f in files %}<option value="{{ f }}">{{ f }}</option>{% endfor %}
              </select>
            </div>
            <div class="col-sm-4"><button class="btn btn-success w-100" type="submit">Build Baseline</button></div>
          </form>
          <div class="mt-2">
            <a class="btn btn-outline-secondary btn-sm" href="{{ url_for('reset_baseline') }}">Reset Baseline</a>
          </div>
          <small class="text-muted">Saves baseline_mean.json (scaled feature mean).</small>
        </div>
      </div>
    </div>

    <div class="col-lg-6">
      <div class="card mb-3">
        <div class="card-header">2) Test / Predict</div>
        <div class="card-body">
          <form method="post" action="{{ url_for('predict') }}" class="row g-3">
            <div class="col-sm-8">
              <select class="form-select" name="filename">
                {% for f in files %}<option value="{{ f }}">{{ f }}</option>{% endfor %}
              </select>
            </div>
            <div class="col-sm-4"><button class="btn btn-warning w-100" type="submit">Run Prediction</button></div>
          </form>
        </div>
      </div>
    </div>
  </div>

  {% if results %}
  <div class="card mb-3">
    <div class="card-header">Results</div>
    <div class="card-body">
      <p>
        <span class="badge rounded-pill bg-primary badge-space">Samples: {{ results.samples }}</span>
        <span class="badge rounded-pill bg-info badge-space">Pred time: {{ results.pred_time }} s</span>
        {% if results.drift is not none %}
          <span class="badge rounded-pill bg-secondary badge-space">Drift: {{ '%.3f'|format(results.drift) }}</span>
        {% endif %}
        {% if results.acc is not none %}
          <span class="badge rounded-pill bg-success badge-space">Acc: {{ results.acc }}</span>
          <span class="badge rounded-pill bg-success badge-space">F1 (Malware): {{ results.f1 }}</span>
        {% else %}
          <span class="badge rounded-pill bg-secondary badge-space">No ground-truth labels</span>
        {% endif %}
      </p>

      <div class="plot-box">
        <img class="img-fluid mb-3" src="{{ url_for('static', filename='plots/pred_counts.png') }}?v={{ cache_bust }}" alt="Prediction counts">
        {% if results.acc is not none %}
          <img class="img-fluid" src="{{ url_for('static', filename='plots/confusion_matrix.png') }}?v={{ cache_bust }}" alt="Confusion matrix">
        {% else %}
          <div class="text-muted">Confusion matrix requires labels in the CSV (label/Label/y/target).</div>
        {% endif %}
      </div>

      {% if results.report %}<pre class="small">{{ results.report }}</pre>{% endif %}
    </div>
  </div>
  {% endif %}

  <p class="text-muted small">Baseline file: baseline/baseline_mean.json</p>
</div>
</body>
</html>
"""

def list_csvs():
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

# ── ROUTES ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML,
        files=list_csvs(),
        n_features=len(SELECTED_FEATURES),
        subset=(SUBSET_N if SUBSET_N else "All"),
        results=None,
        cache_bust=str(int(time.time()))
    )

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("Select a CSV to upload.")
        return redirect(url_for("index"))
    try:
        saved = safe_save_upload(request.files["file"])
        flash(f"Uploaded: {os.path.basename(saved)}")
    except Exception as e:
        flash(f"Upload failed: {e}")
    return redirect(url_for("index"))

@app.route("/baseline", methods=["POST"])
def build_baseline():
    fname = request.form.get("filename")
    if not fname:
        flash("Pick a CSV for baseline.")
        return redirect(url_for("index"))
    path = os.path.join(DATA_DIR, fname)
    try:
        df  = pd.read_csv(path, low_memory=False)
        X   = clean_and_align(df, fname)
        Xs  = scaler.transform(X)
        save_baseline_vector(Xs)
        flash(f"Baseline updated from {fname} ({len(X)} samples).")
    except Exception as e:
        flash(f"Baseline failed: {e}")
    return redirect(url_for("index"))

@app.route("/reset_baseline", methods=["GET"])
def reset_baseline():
    try:
        if os.path.exists(BASELINE_VEC_PATH):
            os.remove(BASELINE_VEC_PATH)
            flash("Baseline reset (baseline_mean.json removed).")
        else:
            flash("No baseline file to remove.")
    except Exception as e:
        flash(f"Reset failed: {e}")
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    fname = request.form.get("filename")
    if not fname:
        flash("Pick a CSV to predict.")
        return redirect(url_for("index"))
    path = os.path.join(DATA_DIR, fname)

    results = {"samples": 0, "pred_time": None, "acc": None, "f1": None, "report": None, "drift": None}

    try:
        df   = pd.read_csv(path, low_memory=False)
        t0   = time.time()
        X    = clean_and_align(df, fname)
        Xs   = scaler.transform(X)
        y_pred = model.predict(Xs)
        pred_time = round(time.time() - t0, 4)

        # plots
        plot_pred_counts(y_pred, os.path.join(PLOTS_DIR, "pred_counts.png"))

        # drift vs baseline
        b = load_baseline_vector()
        results["drift"] = drift_value(Xs, b)

        # metrics if labels present
        y_true, mode = infer_labels(df, fname)
        if y_true is not None:
            y_true = y_true.loc[X.index].astype(int)
            rep = classification_report(y_true, y_pred, target_names=["Benign","Malware"], digits=4)
            cm  = confusion_matrix(y_true, y_pred)
            acc = (cm.trace() / cm.sum()) if cm.sum() else 0.0
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            precision = tp/(tp+fp) if (tp+fp) else 0.0
            recall    = tp/(tp+fn) if (tp+fn) else 0.0
            f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
            plot_cm(y_true, y_pred, "Confusion Matrix", os.path.join(PLOTS_DIR, "confusion_matrix.png"))
            results.update({"samples": int(len(X)), "pred_time": pred_time,
                            "acc": round(acc,4), "f1": round(f1,4), "report": rep})
        else:
            counts = pd.Series(y_pred).value_counts().sort_index().to_dict()
            results.update({"samples": int(len(X)), "pred_time": pred_time,
                            "report": f"No labels (mode={mode}). Prediction counts: {counts}"})
        flash(f"Prediction complete for folder: {os.path.dirname(path) or path}")
    except Exception as e:
        flash(f"Predict failed: {e}")
        results = None

    return render_template_string(
        HTML,
        files=list_csvs(),
        n_features=len(SELECTED_FEATURES),
        subset=(SUBSET_N if SUBSET_N else "All"),
        results=results,
        cache_bust=str(int(time.time()))  # cache-bust plot images
    )

def maybe_start_ngrok():
    if not ENABLE_NGROK:
        return
    try:
        from pyngrok import ngrok, conf
        if NGROK_AUTHTOKEN:
            conf.get_default().auth_token = NGROK_AUTHTOKEN
        public_url = ngrok.connect(5000, "http").public_url
        print(f" * ngrok tunnel: {public_url}")
    except Exception as e:
        print(f"ngrok failed: {e}")

if __name__ == "__main__":
    print("Starting app…")
    maybe_start_ngrok()
    app.run(host="0.0.0.0", port=5000, debug=False)
