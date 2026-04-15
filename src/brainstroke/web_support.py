from __future__ import annotations

import json
import os
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from .core.config import CLASSES, WEB_DIR
from .inference import predict_single_image


MODEL_LABELS = {
    "ensemble": "Ensemble",
    "sg_densenet201_se": "SG-DenseNet201+SE",
    "sg_mobilenet_v2": "SG-MobileNetV2",
    "sg_resnet50": "SG-ResNet50",
    "sg_xception": "SG-Xception",
    "sg_convnext_small": "SG-ConvNeXtSmall",
    "all": "All Models",
}


PRIMARY_RESULT_KEYS = {
    "ensemble": "Ensemble",
    "sg_densenet201_se": "sg_densenet201_se",
    "sg_mobilenet_v2": "sg_mobilenet_v2",
    "sg_resnet50": "sg_resnet50",
    "sg_xception": "sg_xception",
    "sg_convnext_small": "sg_convnext_small",
}


PRECAUTIONS = {
    "Normal": {
        "headline": "The scan looks closer to a normal pattern in this model review.",
        "urgency": "Low immediate alert from this model output.",
        "actions": [
            "Keep regular blood pressure, blood sugar, cholesterol, and diabetes follow-up checks.",
            "Continue healthy routines like enough sleep, hydration, walking, and avoiding smoking.",
            "If symptoms like facial droop, arm weakness, sudden confusion, or speech trouble appear, seek emergency care even if this result looks normal.",
            "Use this result as a screening aid only and confirm with a clinician if symptoms or risk factors are present.",
        ],
        "response_steps": [
            "Keep observing symptoms and general health.",
            "Maintain routine neurological or physician follow-up if already advised.",
            "Act immediately if any sudden stroke warning signs begin.",
        ],
    },
    "Ischemia": {
        "headline": "Possible ischemic stroke pattern detected.",
        "urgency": "High priority. Fast medical evaluation matters.",
        "actions": [
            "Seek emergency medical care immediately, especially if symptoms started recently.",
            "Note the last known normal time because treatment decisions depend on timing.",
            "Do not drive yourself if symptoms are active; use emergency support if available.",
            "Keep a list of current medicines, especially blood thinners, ready for clinicians.",
            "Keep the patient resting safely with the head supported while waiting for medical help.",
            "Do not give food, water, or oral medicines if swallowing is difficult or the person is drowsy.",
        ],
        "response_steps": [
            "Call emergency support without delay.",
            "Track symptom start time and important medical history.",
            "Get urgent hospital imaging and stroke-team evaluation.",
        ],
    },
    "Bleeding": {
        "headline": "Possible bleeding-related stroke pattern detected.",
        "urgency": "Critical priority. This needs urgent hospital evaluation.",
        "actions": [
            "Go to emergency care immediately and avoid delaying for home observation.",
            "Avoid taking aspirin or similar medicines unless a clinician specifically advises it.",
            "Monitor for worsening headache, vomiting, confusion, weakness, or reduced alertness.",
            "Keep the patient resting safely and arrange urgent transport to a hospital.",
            "If the person becomes less responsive, has seizures, or breathing changes, treat it as an extreme emergency.",
            "Do not allow strenuous movement or unnecessary walking while waiting for urgent care.",
        ],
        "response_steps": [
            "Move to emergency care immediately.",
            "Avoid self-medication unless prescribed by a clinician.",
            "Watch closely for rapid worsening while transport is arranged.",
        ],
    },
}


@dataclass
class ReportAssets:
    classification: Optional[str]
    explainability: Optional[str]
    metrics: Optional[str]


WEB_INFER_DIR = WEB_DIR / "inference"
WEB_INFER_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = WEB_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_primary_result(results: Dict[str, torch.Tensor], model_choice: str):
    if model_choice in results:
        return model_choice, results[model_choice]
    key = PRIMARY_RESULT_KEYS.get(model_choice, "Ensemble")
    if key not in results:
        key = "Ensemble" if "Ensemble" in results else list(results.keys())[0]
    return key, results[key]


def _build_asset_paths(image_path: str) -> ReportAssets:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    assets = {
        "classification": WEB_INFER_DIR / f"inference_{stem}_classification.png",
        "explainability": WEB_INFER_DIR / f"inference_{stem}_explainability.png",
        "metrics": WEB_INFER_DIR / f"inference_{stem}_metrics.png",
    }
    return ReportAssets(
        classification=str(assets["classification"]) if assets["classification"].exists() else None,
        explainability=str(assets["explainability"]) if assets["explainability"].exists() else None,
        metrics=str(assets["metrics"]) if assets["metrics"].exists() else None,
    )


def _stroke_stage(label: str, confidence: float) -> str:
    if label == "Normal":
        return "Monitoring"
    if confidence >= 0.85:
        return "Critical Review"
    if confidence >= 0.65:
        return "Urgent Review"
    return "Clinical Review"


def generate_web_report(
    image_path: str,
    models,
    model_choice: str,
) -> dict:
    results = predict_single_image(image_path, models, model_choice=model_choice, save_outputs=True, output_dir=WEB_INFER_DIR)
    primary_key, primary_probs = _resolve_primary_result(results, model_choice)
    primary_probs = primary_probs.tolist()

    top_idx = max(range(len(primary_probs)), key=primary_probs.__getitem__)
    top_label = CLASSES[top_idx]
    top_confidence = primary_probs[top_idx]
    all_models = []
    for model_name, probs in results.items():
        probs_list = probs.tolist()
        idx = max(range(len(probs_list)), key=probs_list.__getitem__)
        all_models.append(
            {
                "name": model_name,
                "prediction": CLASSES[idx],
                "confidence": probs_list[idx],
                "probabilities": [
                    {"label": label, "value": value}
                    for label, value in zip(CLASSES, probs_list)
                ],
            }
        )

    precautions = PRECAUTIONS[top_label]
    assets = _build_asset_paths(image_path)

    return {
        "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "selected_model": MODEL_LABELS.get(model_choice, model_choice),
        "selected_model_key": primary_key,
        "prediction": top_label,
        "confidence": top_confidence,
        "review_stage": _stroke_stage(top_label, top_confidence),
        "summary": precautions["headline"],
        "urgency": precautions["urgency"],
        "probabilities": [
            {"label": label, "value": value}
            for label, value in zip(CLASSES, primary_probs)
        ],
        "all_models": all_models,
        "precautions": precautions["actions"],
        "response_steps": precautions["response_steps"],
        "assets": assets,
    }


# Remaining PDF generation code unchanged

def save_web_report(report: dict, image_path: str) -> str:
    report_id = f"report_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    payload = dict(report)
    payload["image_path"] = str(image_path)
    payload["assets"] = {
        "classification": report["assets"].classification,
        "explainability": report["assets"].explainability,
        "metrics": report["assets"].metrics,
    }
    out_path = REPORTS_DIR / f"{report_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_id


def load_web_report(report_id: str) -> dict:
    report_path = REPORTS_DIR / f"{report_id}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_id}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assets = payload.get("assets", {})
    payload["assets"] = ReportAssets(
        classification=assets.get("classification"),
        explainability=assets.get("explainability"),
        metrics=assets.get("metrics"),
    )
    return payload


def _draw_wrapped_text(pdf, text: str, x: float, y: float, width: float, font_name="Helvetica", font_size=11, leading=15, color=colors.black):
    words = text.split()
    line = ""
    pdf.setFont(font_name, font_size)
    pdf.setFillColor(color)
    for word in words:
        trial = f"{line} {word}".strip()
        if stringWidth(trial, font_name, font_size) <= width:
            line = trial
        else:
            pdf.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _draw_image(pdf, image_path: str, x: float, y_top: float, max_width: float, max_height: float):
    if not image_path or not os.path.exists(image_path):
        return y_top
    image = ImageReader(image_path)
    iw, ih = image.getSize()
    scale = min(max_width / iw, max_height / ih)
    draw_w = iw * scale
    draw_h = ih * scale
    y = y_top - draw_h
    pdf.drawImage(image, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
    return y - 14


def build_report_pdf(report: dict) -> BytesIO:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    def new_page():
        nonlocal y
        pdf.showPage()
        y = height - margin

    pdf.setTitle("Brain Stroke Detection Report")
    pdf.setFillColor(colors.HexColor("#0b5d5b"))
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(margin, y, "Brain Stroke Detection Report")
    y -= 28

    pdf.setFillColor(colors.HexColor("#596273"))
    pdf.setFont("Helvetica", 11)
    pdf.drawString(margin, y, f"Generated: {report['generated_at']}")
    y -= 16
    pdf.drawString(margin, y, f"Model used: {report['selected_model']}")
    y -= 26

    pdf.setFillColor(colors.HexColor("#18212d"))
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, y, f"Prediction: {report['prediction']}")
    y -= 20
    pdf.setFont("Helvetica", 11)
    y = _draw_wrapped_text(pdf, report["summary"], margin, y, width - (2 * margin), color=colors.HexColor("#18212d"))
    y -= 8

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, f"Confidence: {report['confidence'] * 100:.1f}%")
    y -= 16
    pdf.drawString(margin, y, f"Review stage: {report['review_stage']}")
    y -= 16
    pdf.drawString(margin, y, f"Urgency: {report['urgency']}")
    y -= 26

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(margin, y, "Probability Breakdown")
    y -= 18
    pdf.setFont("Helvetica", 11)
    for item in report["probabilities"]:
        pdf.drawString(margin, y, f"{item['label']}: {item['value'] * 100:.1f}%")
        bar_x = margin + 125
        bar_w = width - bar_x - margin
        pdf.setFillColor(colors.HexColor("#e7ecef"))
        pdf.roundRect(bar_x, y - 2, bar_w, 10, 5, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor("#0b5d5b"))
        pdf.roundRect(bar_x, y - 2, bar_w * item["value"], 10, 5, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        y -= 18
    y -= 12

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(margin, y, "Immediate Response Steps")
    y -= 18
    pdf.setFont("Helvetica", 11)
    for index, step in enumerate(report["response_steps"], 1):
        y = _draw_wrapped_text(pdf, f"{index}. {step}", margin, y, width - (2 * margin))
        y -= 4
    y -= 10

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(margin, y, "Precautions")
    y -= 18
    pdf.setFont("Helvetica", 11)
    for index, item in enumerate(report["precautions"], 1):
        if y < 110:
            new_page()
        y = _draw_wrapped_text(pdf, f"{index}. {item}", margin, y, width - (2 * margin))
        y -= 4

    new_page()
    pdf.setFillColor(colors.HexColor("#18212d"))
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, y, "Visual Report")
    y -= 22

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Uploaded Scan")
    y -= 10
    y = _draw_image(pdf, report["image_path"], margin, y, width - (2 * margin), 180)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Classification Output")
    y -= 10
    y = _draw_image(pdf, report["assets"].classification, margin, y, width - (2 * margin), 180)

    if y < 220:
        new_page()

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Explainability Output")
    y -= 10
    y = _draw_image(pdf, report["assets"].explainability, margin, y, width - (2 * margin), 220)

    if y < 220:
        new_page()

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Metrics Output")
    y -= 10
    _draw_image(pdf, report["assets"].metrics, margin, y, width - (2 * margin), 220)

    pdf.save()
    buffer.seek(0)
    return buffer
