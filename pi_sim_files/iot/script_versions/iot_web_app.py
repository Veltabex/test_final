#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, time, uuid, traceback
from datetime import datetime
from typing import Tuple, Dict, Any, List

from flask import (
    Flask, request, redirect, url_for, send_from_directory,
    render_template_string, flash
)

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless for Raspberry Pi
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# -------------------- CONFIG (edit if needed) --------------------
MODEL_PATH  = "/home/pi/Desktop/iot/model/iot_malware_svm_subset_model.pkl"
SCALER_PATH = "/home/pi/Desktop/iot/model/iot_malware_svm_subset_scaler.pkl"

BASE_OUTPUT_DIR = "/home/pi/Desktop/iot/outputs"
UPLOAD_DIR      = "/home/pi/Desktop/iot/uploads"
ALLOWED_EXT     = {'.csv'}

# Map filename suffix → binary label (Benign=0, Malware=1)
FILENAME_LABEL_MAP = {'benign': 0, 'mirai': 1, 'bashlite': 1}
LABEL_ORDER        = ["Benign", "Malware"]
# ---------------------------------------------------------------

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "iot-" + uuid.uuid4().hex  # minimal CSRF/session

# -------------------- Utilities --------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def load_model_scaler() -> Tuple[Any, Any, str]:
    """Load SVM model and scaler; return (model, scaler, msg)."""
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model not found: {MODEL_PATH}"
    if not os.path.exists(SCALER_PATH):
        return None, None, f"Scaler not found: {SCALER_PATH}"
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, ""
    except Exception as e:
        return None, None, f"Failed to load model/scaler: {e}"

def label_from_filename(fname: str) -> int:
    """Infer label using filename (…_benign / …_Mirai / …_BASHLITE)."""
    base = os.path.basename(fname)
    m = re.search(r'(benign|mirai|bashlite)', base, flags=re.IGNORECASE)
    if not m:
        # default benign if unknown
        return 0
    return FILENAME_LABEL_MAP[m.group(1).lower()]

def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # keep numeric columns only; drop non-numerics except 'label' (if present)
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    drop_cols = [c for c in non_num if c.lower() != 'label']
    X = df.drop(columns=drop_cols + (['label'] if 'label' in df.columns else []), errors='ignore')
    y = df['label'] if 'label' in df.columns else None
    return X, y, drop_cols

def ensure_results_dir(tag: str=None) -> str:
    d = os.path.join(BASE_OUTPUT_DIR, tag or now_tag())
    os.makedirs(d, exist_ok=True)
    return d

def plot_and_save_confusion_matrix(cm: np.ndarray, outdir: str, title: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path

def plot_and_save_roc(y_true: np.ndarray, y_prob: np.ndarray, outdir: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1], ls="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, "roc_curve.png")
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path, roc_auc

def evaluate_dataframe(df: pd.DataFrame, model, scaler, results_dir: str) -> Dict[str, Any]:
    """Scale, predict, compute metrics; save outputs; return summary dict."""
    # infer labels from filename if missing
    if 'label' not in df.columns and 'source_file' in df.columns:
        df['label'] = df['source_file'].apply(label_from_filename)
    elif 'label' not in df.columns:
        df['label'] = 0  # default benign if nothing provided

    X, y, dropped = prepare_xy(df)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # proba for SVC might be disabled; handle gracefully
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_scaled)[:, 1]
        except Exception:
            # fallback: distance to decision function
            y_prob = getattr(model, "decision_function")(X_scaled)
            # scale to [0,1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
    else:
        y_prob = getattr(model, "decision_function")(X_scaled)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)

    # metrics
    report = classification_report(y, y_pred, target_names=LABEL_ORDER, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(results_dir, "classification_report.csv")
    report_df.to_csv(report_csv)

    cm = confusion_matrix(y, y_pred)
    cm_png = plot_and_save_confusion_matrix(cm, results_dir, "Overall Confusion Matrix")
    roc_png, roc_auc = plot_and_save_roc(y, y_prob, results_dir)

    # save predictions
    out_csv = os.path.join(results_dir, "predictions.csv")
    out_df = df.copy()
    out_df["predicted"] = y_pred
    out_df.to_csv(out_csv, index=False)

    return {
        "report_csv": report_csv,
        "cm_png": cm_png,
        "roc_png": roc_png,
        "roc_auc": float(roc_auc),
        "predictions_csv": out_csv,
        "dropped_columns": dropped,
        "n_rows": int(len(df))
    }

def read_folder(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSV files found in the folder.")
    frames = []
    for fname in files:
        p = os.path.join(folder_path, fname)
        df = pd.read_csv(p)
        df["source_file"] = fname
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# -------------------- Routes --------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IoT Malware Detection – Baseline & Web UI</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:24px; color:#0a2540;}
    .card{border:1px solid #d0e3ff; border-radius:10px; padding:16px; margin-bottom:16px; background:#f8fbff;}
    h1{margin-bottom:8px}
    label{font-weight:600}
    input[type=text]{width:420px; padding:7px}
    .btn{background:#1a73e8; color:#fff; padding:8px 14px; border:none; border-radius:6px; cursor:pointer}
    .btn.red{background:#e54;}
    .row{display:flex; gap:18px; flex-wrap:wrap}
    .muted{color:#6b7a90; font-size:0.95em}
    img{max-width:480px; border:1px solid #eee; border-radius:8px}
    .ok{color:#0a8}
    .warn{color:#d9534f}
    .tag{display:inline-block;background:#e6f0ff;color:#114;padding:2px 8px;border-radius:999px;margin-left:6px}
  </style>
</head>
<body>
  <h1>IoT Malware Detection Web Console<span class="tag">SVM</span></h1>
  <p class="muted">Model: {{ model_path }} &nbsp; | &nbsp; Scaler: {{ scaler_path }}</p>

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for m in messages %}<p class="warn">{{ m }}</p>{% endfor %}
    {% endif %}
  {% endwith %}

  <div class="card">
    <h3>1) Run Baseline on a Folder</h3>
    <form method="post" action="{{ url_for('run_baseline') }}">
      <label>Folder path with CSVs:</label><br>
      <input type="text" name="folder" value="/home/pi/Desktop/iot/data">
      <button class="btn" type="submit">Run Baseline</button>
    </form>
    <p class="muted">We’ll scan all <code>.csv</code> files, infer labels from filenames (benign/Mirai/BASHLITE), compute metrics, and save results.</p>
  </div>

  <div class="card">
    <h3>2) Upload a CSV</h3>
    <form method="post" action="{{ url_for('upload_csv') }}" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv">
      <button class="btn" type="submit">Analyze File</button>
    </form>
  </div>

  {% if results %}
    <div class="card">
      <h3>Latest Results ({{ results['n_rows'] }} rows)</h3>
      <p>
        <b>ROC AUC:</b> {{ "%.4f"|format(results['roc_auc']) }} &nbsp; | &nbsp;
        <a href="{{ url_for('static_file', path=results['report_csv_rel']) }}">Download Report CSV</a> &nbsp; | &nbsp;
        <a href="{{ url_for('static_file', path=results['predictions_csv_rel']) }}">Download Predictions</a>
      </p>
      <div class="row">
        <div><img src="{{ url_for('static_file', path=results['cm_png_rel']) }}" alt="Confusion Matrix"></div>
        <div><img src="{{ url_for('static_file', path=results['roc_png_rel']) }}" alt="ROC Curve"></div>
      </div>
      {% if results['dropped_columns'] %}
      <p class="muted">Dropped non-numeric columns: {{ results['dropped_columns'] }}</p>
      {% endif %}
      <p class="ok">Saved to: {{ results_dir }}</p>
    </div>
  {% endif %}

  <div class="card">
    <h3>Notes</h3>
    <ul>
      <li>Labels inferred from filename keywords: <code>benign</code>, <code>Mirai</code>, <code>BASHLITE</code>.</li>
      <li>Numeric-only features are fed to the saved scaler & SVM model.</li>
      <li>Outputs saved in a timestamped folder under <code>{{ base_output }}</code>.</li>
    </ul>
  </div>
</body>
</html>
"""

def rel_from_base(path: str) -> str:
    # serve any file inside BASE_OUTPUT_DIR via /files/<path>
    return os.path.relpath(path, BASE_OUTPUT_DIR)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        INDEX_HTML,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        base_output=BASE_OUTPUT_DIR,
        results=None,
        results_dir="",
    )

@app.route("/run_baseline", methods=["POST"])
def run_baseline():
    folder = request.form.get("folder", "").strip()
    model, scaler, msg = load_model_scaler()
    if msg:
        flash(msg); return redirect(url_for("index"))
    try:
        df = read_folder(folder)
        results_dir = ensure_results_dir()
        summary = evaluate_dataframe(df, model, scaler, results_dir)
        # make relative paths for serving
        summary["report_csv_rel"]     = rel_from_base(summary["report_csv"])
        summary["predictions_csv_rel"]= rel_from_base(summary["predictions_csv"])
        summary["cm_png_rel"]         = rel_from_base(summary["cm_png"])
        summary["roc_png_rel"]        = rel_from_base(summary["roc_png"])

        return render_template_string(
            INDEX_HTML,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            base_output=BASE_OUTPUT_DIR,
            results=summary,
            results_dir=results_dir
        )
    except Exception as e:
        traceback.print_exc()
        flash(f"Baseline run failed: {e}")
        return redirect(url_for("index"))

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file provided."); return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only .csv files are allowed."); return redirect(url_for("index"))

    save_path = os.path.join(UPLOAD_DIR, f"{now_tag()}_{file.filename}")
    file.save(save_path)

    model, scaler, msg = load_model_scaler()
    if msg:
        flash(msg); return redirect(url_for("index"))
    try:
        df = pd.read_csv(save_path)
        df["source_file"] = os.path.basename(save_path)

        results_dir = ensure_results_dir()
        summary = evaluate_dataframe(df, model, scaler, results_dir)
        summary["report_csv_rel"]      = rel_from_base(summary["report_csv"])
        summary["predictions_csv_rel"] = rel_from_base(summary["predictions_csv"])
        summary["cm_png_rel"]          = rel_from_base(summary["cm_png"])
        summary["roc_png_rel"]         = rel_from_base(summary["roc_png"])

        return render_template_string(
            INDEX_HTML,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            base_output=BASE_OUTPUT_DIR,
            results=summary,
            results_dir=results_dir
        )
    except Exception as e:
        traceback.print_exc()
        flash(f"Upload analysis failed: {e}")
        return redirect(url_for("index"))

@app.route("/files/<path:path>")
def static_file(path):
    # Serve any file saved under BASE_OUTPUT_DIR
    return send_from_directory(BASE_OUTPUT_DIR, path)

if __name__ == "__main__":
    print(f"[INFO] Starting web app… Model: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=False)
