"""
Plotter Web Server
Flask API: upload image → process → return SVG preview + download
"""

import os
import uuid
import json
import base64
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from processor import process_image, MODES

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ─────────────────────────────────────────────────────────────────────────────
# Default params per mode (sent to frontend for UI defaults)
# ─────────────────────────────────────────────────────────────────────────────

MODE_DEFAULTS = {
    "contour": {
        "blur_radius":  5,
        "canny_low":    40,
        "canny_high":   120,
        "simplify":     1.5,
        "min_length":   10,
        "stroke_width": 0.7,
    },
    "hatching": {
        "angle":        45,
        "line_spacing": 6,
        "cross_hatch":  False,
        "stroke_width": 0.5,
        "invert":       False,
        "contour_mix":  True,
    },
    "stippling": {
        "n_points":     3000,
        "dot_len":      2.0,
        "blur_radius":  9,
        "stroke_width": 0.6,
    },
    "flowfield": {
        "n_lines":      600,
        "step_size":    2.0,
        "max_steps":    80,
        "blur_radius":  11,
        "stroke_width": 0.4,
        "perpendicular": True,
    },
}

# Human-readable param definitions for the UI
PARAM_META = {
    "contour": [
        {"key": "blur_radius",  "label": "Размытие",         "type": "range", "min": 1,  "max": 21, "step": 2,   "tip": "Сглаживает шум перед поиском краёв"},
        {"key": "canny_low",    "label": "Порог (слабые)",   "type": "range", "min": 5,  "max": 150,"step": 5,   "tip": "Нижний порог детектора Canny"},
        {"key": "canny_high",   "label": "Порог (сильные)",  "type": "range", "min": 30, "max": 300,"step": 10,  "tip": "Верхний порог детектора Canny"},
        {"key": "simplify",     "label": "Упрощение линий",  "type": "range", "min": 0,  "max": 8,  "step": 0.5, "tip": "RDP epsilon: выше = меньше точек"},
        {"key": "min_length",   "label": "Мин. длина",       "type": "range", "min": 2,  "max": 50, "step": 1,   "tip": "Отбросить контуры короче N пикселей"},
        {"key": "stroke_width", "label": "Толщина линии",    "type": "range", "min": 0.2,"max": 3,  "step": 0.1, "tip": "Толщина пера в SVG"},
    ],
    "hatching": [
        {"key": "angle",        "label": "Угол штриховки °", "type": "range", "min": 0,  "max": 180,"step": 5,   "tip": "0 = горизонтальные, 45 = диагональные"},
        {"key": "line_spacing", "label": "Шаг строк",        "type": "range", "min": 2,  "max": 24, "step": 1,   "tip": "Расстояние между рядами штрихов"},
        {"key": "cross_hatch",  "label": "Крест-накрест",    "type": "bool",  "tip": "Добавить второй слой под 90°"},
        {"key": "contour_mix",  "label": "Контуры поверх",   "type": "bool",  "tip": "Добавить линии краёв объектов"},
        {"key": "invert",       "label": "Инвертировать",    "type": "bool",  "tip": "Светлые линии на тёмном фоне"},
        {"key": "stroke_width", "label": "Толщина линии",    "type": "range", "min": 0.2,"max": 3,  "step": 0.1, "tip": "Толщина пера в SVG"},
    ],
    "stippling": [
        {"key": "n_points",     "label": "Кол-во точек",     "type": "range", "min": 200,"max": 8000,"step": 100, "tip": "Больше точек = больше деталей"},
        {"key": "dot_len",      "label": "Длина штриха",     "type": "range", "min": 0,  "max": 8,  "step": 0.5, "tip": "0 = точка, >0 = мини-штрих"},
        {"key": "blur_radius",  "label": "Размытие",         "type": "range", "min": 1,  "max": 25, "step": 2,   "tip": "Сглаживание карты вероятности"},
        {"key": "stroke_width", "label": "Толщина линии",    "type": "range", "min": 0.2,"max": 3,  "step": 0.1, "tip": "Толщина пера в SVG"},
    ],
    "flowfield": [
        {"key": "n_lines",      "label": "Кол-во линий",     "type": "range", "min": 50, "max": 2000,"step": 50,  "tip": "Больше линий = больше покрытие"},
        {"key": "step_size",    "label": "Шаг частицы",      "type": "range", "min": 0.5,"max": 8,  "step": 0.5, "tip": "Длина шага за итерацию"},
        {"key": "max_steps",    "label": "Макс. шагов",      "type": "range", "min": 10, "max": 300,"step": 10,  "tip": "Максимальная длина одной линии"},
        {"key": "blur_radius",  "label": "Гладкость потока", "type": "range", "min": 1,  "max": 31, "step": 2,   "tip": "Размытие перед вычислением градиента"},
        {"key": "perpendicular","label": "Перпенд. к краям", "type": "bool",  "tip": "Следовать вдоль, а не поперёк контуров"},
        {"key": "stroke_width", "label": "Толщина линии",    "type": "range", "min": 0.2,"max": 3,  "step": 0.1, "tip": "Толщина пера в SVG"},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/modes", methods=["GET"])
def get_modes():
    """Return available modes with their default params and UI metadata."""
    return jsonify({
        "modes": list(MODES.keys()),
        "defaults": MODE_DEFAULTS,
        "meta": PARAM_META,
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload an image. Returns image_id for subsequent processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format. Use: {ALLOWED_EXTS}"}), 400

    image_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{image_id}{ext}"

    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large (max 20 MB)"}), 413

    f.save(save_path)

    # Return small base64 thumbnail for preview
    import cv2
    import numpy as np
    img = cv2.imread(str(save_path))
    th = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
    _, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 75])
    thumb_b64 = base64.b64encode(buf).decode()

    return jsonify({
        "image_id": image_id,
        "filename": f.filename,
        "thumbnail": f"data:image/jpeg;base64,{thumb_b64}",
        "size": [img.shape[1], img.shape[0]],
    })


@app.route("/api/process", methods=["POST"])
def process():
    """Process uploaded image with chosen mode and params. Returns SVG."""
    data = request.json or {}
    image_id = data.get("image_id")
    mode     = data.get("mode", "contour")
    params   = data.get("params", {})

    if not image_id:
        return jsonify({"error": "image_id required"}), 400

    # Find the uploaded file (any allowed extension)
    src = None
    for ext in ALLOWED_EXTS:
        candidate = UPLOAD_DIR / f"{image_id}{ext}"
        if candidate.exists():
            src = candidate
            break
    if src is None:
        return jsonify({"error": "Image not found. Upload first."}), 404

    # Output path
    out_id  = str(uuid.uuid4())[:8]
    out_svg = OUTPUT_DIR / f"{image_id}_{mode}_{out_id}.svg"
    params["output"] = str(out_svg)

    t0 = time.time()
    try:
        svg_path = process_image(str(src), mode, params)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    elapsed = round(time.time() - t0, 2)

    # Read SVG and return inline
    svg_content = Path(svg_path).read_text(encoding="utf-8")

    # Count strokes (rough)
    stroke_count = svg_content.count("<polyline") + svg_content.count("<line")

    return jsonify({
        "svg": svg_content,
        "svg_file": out_svg.name,
        "elapsed_s": elapsed,
        "stroke_count": stroke_count,
        "mode": mode,
    })


@app.route("/api/download/<filename>", methods=["GET"])
def download(filename):
    """Download a previously generated SVG file."""
    path = OUTPUT_DIR / filename
    if not path.exists() or path.suffix != ".svg":
        return jsonify({"error": "Not found"}), 404
    return send_file(str(path), mimetype="image/svg+xml",
                     as_attachment=True, download_name=filename)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Plotter Web Server")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
