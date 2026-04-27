from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[2]

default_artifacts = PROJECT_ROOT / "artifacts"
default_checkpoints = default_artifacts / "train" / "checkpoints"
if not default_checkpoints.exists() and default_artifacts.exists():
    os.environ.setdefault("STROKE_PRETRAINED_DIR", str(default_artifacts / "train" / "checkpoints"))
    os.environ.setdefault("STROKE_MODEL_DIR", str(default_artifacts / "train" / "checkpoints"))

sys.path.append(str(PROJECT_ROOT / "src"))

from brainstroke.core.config import WEB_DIR  # noqa: E402
from brainstroke.model_io import load_models  # noqa: E402
from brainstroke.web_support import (  # noqa: E402
    MODEL_LABELS,
    build_report_pdf,
    generate_web_report,
    load_web_report,
    save_web_report,
)


UPLOAD_DIR = PROJECT_ROOT / "artifacts" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
WEB_INFER_DIR = WEB_DIR / "inference"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

_MODELS = None


def get_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models()
    return _MODELS


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage) -> Path:
    ext = Path(file_storage.filename).suffix.lower()
    name = secure_filename(Path(file_storage.filename).stem) or "scan"
    out_path = UPLOAD_DIR / f"{name}_{uuid4().hex[:8]}{ext}"
    file_storage.save(out_path)
    return out_path


@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    error = None
    selected_model = request.form.get("model", "ensemble")
    preview_path = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or uploaded.filename == "":
            error = "Please choose an image file first."
        elif not allowed_file(uploaded.filename):
            error = "Supported files: PNG, JPG, JPEG, and BMP."
        else:
            try:
                image_path = save_upload(uploaded)
                preview_path = url_for("serve_uploaded_file", filename=image_path.name)
                report = generate_web_report(str(image_path), get_models(), selected_model)
                report["report_id"] = save_web_report(report, str(image_path))
                report["preview_url"] = preview_path
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        model_options=MODEL_LABELS,
        selected_model=selected_model,
        report=report,
        error=error,
        infer_dir=str(WEB_INFER_DIR),
    )


@app.route("/uploads/<path:filename>")
def serve_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        abort(404)
    return send_file(file_path)


@app.route("/reports/<path:kind>/<path:filename>")
def serve_report_file(kind: str, filename: str):
    if kind not in {"classification", "segmentation", "explainability", "metrics"}:
        abort(404)
    file_path = WEB_INFER_DIR / filename
    if not file_path.exists():
        abort(404)
    return send_file(file_path)


@app.route("/download-report/<report_id>")
def download_report(report_id: str):
    try:
        report = load_web_report(report_id)
    except FileNotFoundError:
        abort(404)
    pdf_buffer = build_report_pdf(report)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"{report_id}.pdf",
        mimetype="application/pdf",
    )


@app.context_processor
def inject_helpers():
    def report_asset_url(asset_path: str | None):
        if not asset_path:
            return None
        file_name = Path(asset_path).name
        if file_name.endswith("_classification.png"):
            kind = "classification"
        elif file_name.endswith("_segmentation.png"):
            kind = "segmentation"
        elif file_name.endswith("_explainability.png"):
            kind = "explainability"
        else:
            kind = "metrics"
        return url_for("serve_report_file", kind=kind, filename=file_name)

    return {"report_asset_url": report_asset_url}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
