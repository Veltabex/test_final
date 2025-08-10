import os
import io
import json
import time
import random
import traceback
import numpy as np
import pandas as pd

from flask import Flask, request, render_template_string, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import joblib
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Config
# -----------------------------
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")
UPLOAD_DIR   = os.environ.get("UPLOAD_DIR",   "./uploads")
PLOTS_DIR    = os.environ.get("PLOTS_DIR",    "./plots")
BASELINE_PATH = os.path.join(ARTIFACT_DIR, "baseline_stats.json")
SUBSET_MAX   = int(os.environ.get("SUBSET_MAX", "5000"))   # cap for random subset
ALLOW_UPLOAD = True  # keep old functionality of browsing/uploading CSVs

# Optional ngrok (do not hardcode your token here)
ENABLE_NGROK = os.environ.get("ENABLE_NGROK", "0") == "1"

# File names (you will place your old model + scaler + feature files under ARTIFACT_DIR)
MODEL_PATH            = os.path.join(ARTIFACT_DIR, "svm_model.pkl")
SCALER_PATH           = os.path.join(ARTIFACT_DIR, "scaler.pkl")
SELECTED_FEATURES_JS  = os.path.join(ARTIFACT_DIR, "selected_features.json")
LABEL_MAP_JS          = os.path.join(ARTIFACT_DIR, "label_map.json")
MANIFEST_JS           = os.path.join(ARTIFACT_DIR, "manifest.json")  # optional metadata

# -----------------------------
# App + FS
# -----------------------------
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")  # replace in production

# Globals loaded once
model = None
scaler = None
selected_features = None
label_map = None
manifest = {}

compat_warnings = []  # we’ll show these on the UI banner

# -----------------------------
# Utils
# -----------------------------
def load_artifacts():
    global model, scaler, selected_features, label_map, manifest, compat_warnings
    compat_warnings.clear()

    # Load JSONs
    def _load_json(path, required=True):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Load core artifacts
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = _load_json(SELECTED_FEATURES_JS, required=True)
    label_map = _load_json(LABEL_MAP_JS, required=True)
    manifest = _load_json(MANIFEST_JS, required=False)

    # Basic compatibility checks
    # 1) selected_features should be list[str]
    if not isinstance(selected_features, list) or not all(isinstance(c, str) for c in selected_features):
        compat_warnings.append("selected_features.json is malformed (should be a list of strings).")

    # 2) model expects same number of features
    if hasattr(model, "n_features_in_"):
        if len(selected_features) != int(model.n_features_in_):
            compat_warnings.append(
                f"Model expects {int(model.n_features_in_)} features, but selected_features.json has {len(selected_features)}."
            )

    # 3) scaler feature size sanity check (best‑effort)
    # StandardScaler doesn’t keep names, but it does have mean_/scale_ length
    try:
        if hasattr(scaler, "mean_"):
            if len(selected_features) != len(scaler.mean_):
                compat_warnings.append(
                    f"Scaler expects {len(scaler.mean_)} features, but selected_features.json has {len(selected_features)}."
                )
    except Exception:
        compat_warnings.append("Could not validate scaler’s expected feature length.")

    # 4) label_map consistency vs model classes_ (binary case)
    try:
        if hasattr(model, "classes_"):
            cls_set = set(model.classes_.tolist())
            lm_set = set(label_map.values())
            # For mapping like {'benign':0,'Mirai':1,'BASHLITE':1}, value set should equal model classes
            if cls_set != lm_set:
                compat_warnings.append(
                    f"label_map values {sorted(lm_set)} do not match model.classes_ {sorted(cls_set)}."
                )
    except Exception:
        compat_warnings.append("Could not validate label_map against model classes_.")
    return True


def pick_subset(df: pd.DataFrame, max_n: int = SUBSET_MAX) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    # sample without replacement for speed
    return df.sample(n=max_n, random_state=42)


def load_csvs_from_folder(folder: str) -> pd.DataFrame:
    frames = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".csv"):
            path = os.path.join(folder, fname)
            df = pd.read_csv(path, low_memory=False)
            df["__source_file__"] = fname
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSVs found in {folder}")
    return pd.concat(frames, ignore_index=True)


def prepare_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    # Keep only the selected features (missing get filled with 0 to avoid hard fails)
    missing = [c for c in features if c not in df.columns]
    if missing:
        # Fill missing columns with zeros (or could use df[missing] = np.nan and impute)
        for c in missing:
            df[c] = 0.0
    X = df[features].copy()
    # Replace inf and NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def compute_baseline_stats(df: pd.DataFrame, features: list[str]) -> dict:
    # Per-feature mean and std from (assumed) benign traffic
    X = prepare_matrix(df, features)
    means = X.mean(axis=0).to_dict()
    stds  = X.std(axis=0).replace(0, 1e-9).to_dict()
    return {"features": features, "means": means, "stds": stds, "created_at": time.time()}


def load_baseline():
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_baseline(stats: dict):
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def drift_score(X: pd.DataFrame, baseline: dict) -> float:
    # Simple mean |z| over all cells
    means = pd.Series(baseline["means"])
    stds  = pd.Series(baseline["stds"]).replace(0, 1e-9)
    aligned = X[baseline["features"]]
    z = (aligned - means) / stds
    return float(np.nanmean(np.abs(z.values)))


def ensure_bool(val):
    return "Yes" if val else "No"


# -----------------------------
# HTML (minimal, clean)
# -----------------------------
TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>IoT Malware Detector (Web UI)</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding: 20px; }
    .badge-pill { border-radius: 999px; }
    .toggle-pill {
      width: 60px; height: 28px; background:#d9e5ff; border-radius:14px; position: relative; cursor: default; margin-right:10px;
    }
    .toggle-dot { width: 24px; height:24px; border-radius:50%; background:#6ea8fe; position:absolute; top:2px; left:2px; transition:left .2s; }
    .toggle-on .toggle-dot { left:34px; background:#2ecc71; }
    .wifi-bar { width: 8px; height: 14px; background:#6ea8fe; margin-right:4px; display:inline-block; opacity:.2; }
    .wifi-on .wifi-bar { opacity:1; }
  </style>
</head>
<body>
<div class="container">
  <h1 class="mb-3">IoT Malware Detector</h1>

  {% if compat_warnings %}
  <div class="alert alert-warning">
    <h5 class="mb-2">⚠ Compatibility warnings</h5>
    <ul class="mb-0">
      {% for w in compat_warnings %}
      <li>{{ w }}</li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="alert alert-info">
        {% for m in messages %} <div>{{ m|safe }}</div> {% endfor %}
      </div>
    {% endif %}
  {% endwith %}

  <div class="mb-3">
    <span class="toggle-pill {{ 'toggle-on' if baseline_loaded else '' }}">
      <span class="toggle-dot"></span>
    </span>
    <span class="wifi {{ 'wifi-on' if baseline_loaded else '' }}">
      <span class="wifi-bar"></span>
      <span class="wifi-bar"></span>
      <span class="wifi-bar"></span>
    </span>
    <strong class="ms-2">{{ 'Baseline Loaded' if baseline_loaded else 'Baseline Not Loaded' }}</strong>
  </div>

  <div class="row g-3">
    <div class="col-md-4">
      <div class="card shadow-sm">
        <div class="card-body">
          <h5>Baseline</h5>
          <p class="mb-2">Compute baseline stats (means/stds) from benign CSVs in a folder.</p>
          <form method="post" action="{{ url_for('baseline') }}">
            <div class="mb-2">
              <label class="form-label">Folder path (on device)</label>
              <input name="folder" class="form-control" placeholder="/home/pi/Desktop/iot/benign_data">
            </div>
            <button class="btn btn-primary">Build Baseline</button>
          </form>
          <form class="mt-2" method="post" action="{{ url_for('reset_baseline') }}">
            <button class="btn btn-outline-secondary">Reset Baseline</button>
          </form>
          <div class="mt-2 small text-muted">
            Features: {{ n_features }} |
            Model classes: {{ model_classes }} |
            Subset max: {{ subset_max }}
          </div>
        </div>
      </div>
    </div>

    <div class="col-md-8">
      <div class="card shadow-sm">
        <div class="card-body">
          <h5>Predict</h5>
          <p class="mb-2">
            Load CSVs from a folder, sample up to {{ subset_max }} rows, scale, predict with SVM, and show metrics.
          </p>
          <form method="post" action="{{ url_for('predict') }}">
            <div class="row g-2">
              <div class="col-md-8">
                <label class="form-label">Folder path (on device)</label>
                <input name="folder" class="form-control" placeholder="/home/pi/Desktop/iot/test_data">
              </div>
              <div class="col-md-4">
                <label class="form-label">Assume label for all rows</label>
                <select name="assume_label" class="form-select">
                  <option value="auto" selected>Auto from filename (Mirai/BASHLITE/benign)</option>
                  <option value="0">Force Benign (0)</option>
                  <option value="1">Force Malware (1)</option>
                </select>
              </div>
            </div>
            <button class="btn btn-success mt-2">Run Prediction</button>
          </form>

          {% if results %}
            <hr/>
            <h6>Results</h6>
            <pre style="white-space:pre-wrap">{{ results }}</pre>
            {% if drift is not none %}
              <div class="mt-2">Drift score vs baseline (mean |z|): <strong>{{ "%.3f"|format(drift) }}</strong></div>
            {% endif %}
          {% endif %}
        </div>
      </div>
    </div>
  </div>

  <div class="mt-4">
    <span class="badge bg-primary badge-pill">Real-time Adaptation (planned)</span>
    <span class="badge bg-info text-dark badge-pill">Resource Optimized</span>
    <span class="badge bg-secondary badge-pill">90–95% (dataset-dependent)</span>
  </div>
</div>
</body>
</html>
"""

def human_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    lines = ["Confusion Matrix (rows=true, cols=pred):", str(cm)]
    return "\n".join(lines)

def auto_label_from_filename(name: str) -> int:
    name_l = name.lower()
    if "mirai" in name_l or "bashlite" in name_l:
        return 1
    return 0

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    baseline_loaded = os.path.exists(BASELINE_PATH)
    classes = getattr(model, "classes_", None)
    return render_template_string(
        TEMPLATE,
        compat_warnings=compat_warnings,
        baseline_loaded=baseline_loaded,
        n_features=len(selected_features) if selected_features else 0,
        model_classes=list(classes) if classes is not None else "unknown",
        subset_max=SUBSET_MAX,
        results=None,
        drift=None
    )

@app.route("/baseline", methods=["POST"])
def baseline():
    try:
        folder = request.form.get("folder", "").strip()
        if not folder:
            flash("Please provide a folder path that contains benign CSV files.")
            return redirect(url_for("index"))

        df = load_csvs_from_folder(folder)
        df = pick_subset(df, SUBSET_MAX)

        # Build baseline from the selected features only
        stats = compute_baseline_stats(df, selected_features)
        save_baseline(stats)
        flash(f"✅ Baseline built from {len(df)} rows. Saved to {BASELINE_PATH}")
    except Exception as e:
        flash(f"❌ Baseline error: {e}")
        app.logger.error("Baseline error:\n" + traceback.format_exc())
    return redirect(url_for("index"))

@app.route("/reset", methods=["POST"])
def reset_baseline():
    try:
        if os.path.exists(BASELINE_PATH):
            os.remove(BASELINE_PATH)
            flash("Baseline reset.")
        else:
            flash("No baseline file to delete.")
    except Exception as e:
        flash(f"❌ Reset error: {e}")
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    results_text = ""
    drift_val = None
    try:
        folder = request.form.get("folder", "").strip()
        assume_label = request.form.get("assume_label", "auto")

        if not folder:
            flash("Please provide a folder path with CSVs for prediction.")
            return redirect(url_for("index"))

        df = load_csvs_from_folder(folder)
        df = pick_subset(df, SUBSET_MAX)

        # Extract labels
        if assume_label == "auto":
            # derive from filename
            labels = []
            for fname in df["__source_file__"].tolist():
                labels.append(auto_label_from_filename(fname))
            y_true = np.array(labels, dtype=int)
        else:
            y_true = np.full(len(df), int(assume_label), dtype=int)

        # Prepare features -> scale
        X = prepare_matrix(df, selected_features)
        X_scaled = scaler.transform(X)

        # Predict
        t0 = time.time()
        y_pred = model.predict(X_scaled)
        dt = time.time() - t0

        # Metrics
        try:
            report = classification_report(y_true, y_pred, target_names=["Benign(0)","Malware(1)"], digits=4)
        except Exception:
            report = classification_report(y_true, y_pred, digits=4)

        results_text += f"Rows used: {len(df)} (subset up to {SUBSET_MAX})\n"
        results_text += f"Inference time: {dt:.4f} s (avg {dt/len(df):.6f} s/sample)\n\n"
        results_text += report + "\n\n"
        results_text += human_confusion(y_true, y_pred) + "\n"

        # Drift vs baseline (if available)
        baseline = load_baseline()
        if baseline:
            drift_val = drift_score(X, baseline)

        flash("✅ Prediction complete.")
    except Exception as e:
        flash(f"❌ Prediction error: {e}")
        app.logger.error("Predict error:\n" + traceback.format_exc())

    baseline_loaded = os.path.exists(BASELINE_PATH)
    classes = getattr(model, "classes_", None)
    return render_template_string(
        TEMPLATE,
        compat_warnings=compat_warnings,
        baseline_loaded=baseline_loaded,
        n_features=len(selected_features) if selected_features else 0,
        model_classes=list(classes) if classes is not None else "unknown",
        subset_max=SUBSET_MAX,
        results=results_text if results_text else None,
        drift=drift_val
    )

# Static plots (if you still save any)
@app.route("/plots/<path:filename>")
def plots(filename):
    return send_from_directory(PLOTS_DIR, filename)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load artifacts once
    try:
        load_artifacts()
        print("✅ Loaded model/scaler/features/labels.")
    except Exception as e:
        print("❌ Failed to load artifacts:", e)
        traceback.print_exc()

    # Optional ngrok
    if ENABLE_NGROK:
        try:
            from pyngrok import ngrok, conf
            # Expect NGROK_AUTHTOKEN in env (run once on device: ngrok config add-authtoken <token>)
            token = os.environ.get("NGROK_AUTHTOKEN")
            if token:
                conf.get_default().auth_token = token
            public_url = ngrok.connect(5000, "http")
            print("🌍 Public URL:", public_url.public_url)
        except Exception as e:
            print("⚠ ngrok not started:", e)

    app.run(host="0.0.0.0", port=5000, debug=False)
