import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.services.layout import detect_layout
from app.services.preprocessing import preprocess_image
from app.services.qwen_vl_ocr import get_qwen_vl_ocr

logger = logging.getLogger(__name__)


def assess_image_quality(image_path: str) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        return {
            "readable": False,
            "blur_score": 0.0,
            "contrast_score": 0.0,
            "resolution": {"width": 0, "height": 0},
            "recommendations": ["request_human_review"],
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_score = float(gray.std())
    recommendations: List[str] = []

    if blur_score < 80:
        recommendations.append("image_may_be_blurry")
    if contrast_score < 35:
        recommendations.append("enhance_contrast")
    if width < 800 or height < 600:
        recommendations.append("low_resolution")

    return {
        "readable": True,
        "blur_score": round(blur_score, 2),
        "contrast_score": round(contrast_score, 2),
        "resolution": {"width": int(width), "height": int(height)},
        "recommendations": recommendations,
    }


def preprocess_with_policy(image_path: str, quality: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    actions = []
    try:
        processed_path = preprocess_image(image_path)
        actions.append({
            "action": "preprocess_image",
            "decision": "applied_existing_pipeline",
            "reason": "agent selected preprocessing before OCR to improve readability",
            "output": processed_path,
        })
        return processed_path, actions
    except Exception as exc:
        fallback_path = str(Path(image_path).with_name(f"agent_original_{Path(image_path).name}"))
        shutil.copyfile(image_path, fallback_path)
        actions.append({
            "action": "preprocess_image",
            "decision": "fallback_to_original",
            "reason": str(exc),
            "output": fallback_path,
        })
        return fallback_path, actions


def run_layout_detection(image_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        regions = detect_layout(image_path)
        return regions, {
            "action": "detect_layout",
            "decision": "completed",
            "region_count": len(regions),
        }
    except Exception as exc:
        image = cv2.imread(image_path)
        if image is None:
            regions = []
        else:
            height, width = image.shape[:2]
            regions = [{"type": "paragraph", "bbox": [0, 0, int(width), int(height)]}]
        return regions, {
            "action": "detect_layout",
            "decision": "fallback_full_page_region",
            "reason": str(exc),
            "region_count": len(regions),
        }


def run_qwen_ocr(image_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    image = cv2.imread(image_path)
    height = width = None
    if image is not None:
        height, width = image.shape[:2]

    qwen_ocr = get_qwen_vl_ocr()
    text, diagram_regions = qwen_ocr.extract_text_from_image(
        image_path,
        image_width=width,
        image_height=height,
    )
    return text, diagram_regions, {
        "action": "qwen_vl_ocr",
        "decision": "completed",
        "characters": len(text or ""),
        "diagram_regions": len(diagram_regions),
    }


def parse_structured_text(text: str) -> List[Dict[str, Any]]:
    structured_json: List[Dict[str, Any]] = []
    for line in (text or "").split("\n"):
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("## "):
            structured_json.append({"type": "heading", "text": clean[3:].strip(), "source": "agent_ocr"})
        elif clean.startswith("### "):
            structured_json.append({"type": "heading", "text": clean[4:].strip(), "source": "agent_ocr"})
        elif clean.startswith("$$") and clean.endswith("$$") and len(clean) > 4:
            structured_json.append({"type": "equation", "latex": clean[2:-2].strip(), "source": "agent_ocr"})
        elif clean.startswith("[[DIAGRAM"):
            structured_json.append({"type": "paragraph", "text": "[Diagram detected in handwritten notes]", "source": "agent_diagram_marker"})
        else:
            structured_json.append({"type": "paragraph", "text": clean, "source": "agent_ocr"})
    if not structured_json:
        structured_json.append({
            "type": "paragraph",
            "text": "[Agentic OCR pipeline executed but no readable text was extracted]",
            "source": "agent_empty_fallback",
        })
    return structured_json


def critique_structured_output(structured_json: List[Dict[str, Any]], quality: Dict[str, Any]) -> Dict[str, Any]:
    text_elements = [item.get("text") or item.get("latex") or "" for item in structured_json]
    total_chars = sum(len(item) for item in text_elements)
    suspicious_items = []

    for index, value in enumerate(text_elements):
        if len(value) <= 1:
            suspicious_items.append({"index": index, "reason": "very_short_text"})
        elif _symbol_noise_ratio(value) > 0.45:
            suspicious_items.append({"index": index, "reason": "high_symbol_noise"})

    human_review_required = bool(suspicious_items)
    if quality.get("blur_score", 1000) < 80 or quality.get("contrast_score", 1000) < 35:
        human_review_required = True

    return {
        "total_elements": len(structured_json),
        "total_characters": total_chars,
        "suspicious_items": suspicious_items[:10],
        "human_review_required": human_review_required,
        "summary": "review_recommended" if human_review_required else "output_accepted_by_agent",
    }


def _symbol_noise_ratio(value: str) -> float:
    if not value:
        return 0.0
    noisy = sum(1 for char in value if not char.isalnum() and not char.isspace() and char not in ".,;:()[]{}+-=*/_$\\")
    return noisy / max(1, len(value))
