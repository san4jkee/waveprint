"""Plotter Web Server — v3"""

import uuid, base64, time
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from processor import process_image, MODES

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
ALLOWED_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}
MAX_SIZE = 20 * 1024 * 1024

# ── Param metadata ────────────────────────────────────────────────────────────

R = "range"; B = "bool"

MODE_DEFAULTS = {
    "contour":   {"blur_radius":5,"canny_low":40,"canny_high":120,"simplify":1.5,"min_length":10,"stroke_width":0.7},
    "hatching":  {"angle":45,"line_spacing":6,"cross_hatch":False,"contour_mix":True,"invert":False,"stroke_width":0.5},
    "stippling": {"n_points":3000,"min_dot_len":4.0,"max_dot_len":12.0,"blur_radius":9,"vary_size":True,"stroke_width":0.7},
    "flowfield": {"n_lines":600,"step_size":2.0,"max_steps":80,"blur_radius":11,"perpendicular":True,"stroke_width":0.4},
    "voronoi":   {"n_seeds":600,"blur_radius":9,"contour_mix":True,"stroke_width":0.5},
    "spiral":    {"turns":60,"base_step":5,"min_step":0.5,"center_x":0.5,"center_y":0.5,"stroke_width":0.4},
    "wave":      {"line_spacing":10,"frequency":5,"amplitude":15,"phase_shift":1.2,"contour_mix":False,"stroke_width":0.5},
    "sketch":    {"n_passes":3,"line_spacing":8,"jitter":3,"blur_radius":5,"stroke_width":0.5},
}

PARAM_META = {
    "contour": [
        {"key":"blur_radius", "label":"Размытие",        "type":R,"min":1,  "max":21, "step":2,  "tip":"Сглаживание шума"},
        {"key":"canny_low",   "label":"Порог мин.",       "type":R,"min":5,  "max":150,"step":5,  "tip":"Нижний порог Canny"},
        {"key":"canny_high",  "label":"Порог макс.",      "type":R,"min":30, "max":300,"step":10, "tip":"Верхний порог Canny"},
        {"key":"simplify",    "label":"Упрощение",        "type":R,"min":0,  "max":8,  "step":0.5,"tip":"RDP: выше = меньше точек"},
        {"key":"min_length",  "label":"Мин. контур",      "type":R,"min":2,  "max":50, "step":1,  "tip":"Удалить короткие контуры"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "hatching": [
        {"key":"angle",       "label":"Угол °",           "type":R,"min":0,  "max":180,"step":5,  "tip":"0=горизонт., 45=диаг."},
        {"key":"line_spacing","label":"Шаг строк",        "type":R,"min":2,  "max":24, "step":1,  "tip":"Расстояние между рядами"},
        {"key":"cross_hatch", "label":"Крест-накрест",    "type":B,"tip":"Добавить слой под 90°"},
        {"key":"contour_mix", "label":"Контуры поверх",   "type":B,"tip":"Наложить края объектов"},
        {"key":"invert",      "label":"Инвертировать",    "type":B,"tip":"Светлые линии на тёмном"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "stippling": [
        {"key":"n_points",    "label":"Кол-во точек",     "type":R,"min":200,"max":8000,"step":100,"tip":"Больше = детальнее"},
        {"key":"min_dot_len", "label":"Мин. длина штриха","type":R,"min":1,  "max":10, "step":0.5,"tip":"Длина в светлых зонах"},
        {"key":"max_dot_len", "label":"Макс. длина",      "type":R,"min":2,  "max":20, "step":0.5,"tip":"Длина в тёмных зонах"},
        {"key":"blur_radius", "label":"Размытие карты",   "type":R,"min":1,  "max":25, "step":2,  "tip":"Сглаживание распределения"},
        {"key":"vary_size",   "label":"Разный размер",    "type":B,"tip":"Тёмные зоны = длиннее штрих"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "flowfield": [
        {"key":"n_lines",     "label":"Кол-во линий",     "type":R,"min":50, "max":2000,"step":50, "tip":"Больше = плотнее"},
        {"key":"step_size",   "label":"Шаг частицы",      "type":R,"min":0.5,"max":8,  "step":0.5,"tip":"Длина шага за итерацию"},
        {"key":"max_steps",   "label":"Макс. шагов",      "type":R,"min":10, "max":300,"step":10, "tip":"Макс. длина линии"},
        {"key":"blur_radius", "label":"Гладкость",        "type":R,"min":1,  "max":31, "step":2,  "tip":"Размытие градиента"},
        {"key":"perpendicular","label":"Вдоль контуров",  "type":B,"tip":"Перпенд. к градиенту"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "voronoi": [
        {"key":"n_seeds",     "label":"Кол-во ячеек",     "type":R,"min":100,"max":2000,"step":50, "tip":"Больше = детальнее"},
        {"key":"blur_radius", "label":"Размытие карты",   "type":R,"min":1,  "max":25, "step":2,  "tip":"Сглаживание вероятности"},
        {"key":"contour_mix", "label":"Контуры поверх",   "type":B,"tip":"Наложить края объектов"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "spiral": [
        {"key":"turns",       "label":"Кол-во витков",    "type":R,"min":10, "max":150,"step":5,  "tip":"Больше = плотнее спираль"},
        {"key":"base_step",   "label":"Шаг (светлые)",    "type":R,"min":1,  "max":20, "step":0.5,"tip":"Расстояние витков в светл. зонах"},
        {"key":"min_step",    "label":"Шаг (тёмные)",     "type":R,"min":0.1,"max":5,  "step":0.1,"tip":"Расстояние витков в тёмн. зонах"},
        {"key":"center_x",    "label":"Центр X",          "type":R,"min":0,  "max":1,  "step":0.05,"tip":"Горизонтальное положение центра"},
        {"key":"center_y",    "label":"Центр Y",          "type":R,"min":0,  "max":1,  "step":0.05,"tip":"Вертикальное положение центра"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "wave": [
        {"key":"line_spacing","label":"Шаг строк",        "type":R,"min":4,  "max":30, "step":1,  "tip":"Расстояние между волнами"},
        {"key":"frequency",   "label":"Частота волн",     "type":R,"min":1,  "max":20, "step":0.5,"tip":"Циклов на ширину изображения"},
        {"key":"amplitude",   "label":"Амплитуда",        "type":R,"min":2,  "max":40, "step":1,  "tip":"Макс. отклонение волны px"},
        {"key":"phase_shift", "label":"Фаза строк",       "type":R,"min":0,  "max":6,  "step":0.1,"tip":"Сдвиг фазы между рядами"},
        {"key":"contour_mix", "label":"Контуры поверх",   "type":B,"tip":"Наложить края объектов"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
    "sketch": [
        {"key":"n_passes",    "label":"Кол-во проходов",  "type":R,"min":1,  "max":5,  "step":1,  "tip":"Слои штриховки под разными углами"},
        {"key":"line_spacing","label":"Шаг строк",        "type":R,"min":4,  "max":20, "step":1,  "tip":"Расстояние между штрихами"},
        {"key":"jitter",      "label":"Дрожание",         "type":R,"min":0,  "max":10, "step":0.5,"tip":"Случайное смещение штрихов"},
        {"key":"blur_radius", "label":"Размытие",         "type":R,"min":1,  "max":15, "step":2,  "tip":"Сглаживание перед штриховкой"},
        {"key":"stroke_width","label":"Толщина линии",    "type":R,"min":0.2,"max":3,  "step":0.1,"tip":"Ширина пера"},
    ],
}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/modes")
def get_modes():
    return jsonify({"modes": list(MODES), "defaults": MODE_DEFAULTS, "meta": PARAM_META})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Use: {ALLOWED_EXTS}"}), 400
    f.seek(0,2); size = f.tell(); f.seek(0)
    if size > MAX_SIZE:
        return jsonify({"error": "File too large (max 20 MB)"}), 413
    image_id  = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{image_id}{ext}"
    f.save(save_path)
    import cv2, numpy as np
    img = cv2.imread(str(save_path))
    th  = cv2.resize(img, (320, max(1, int(320*img.shape[0]/img.shape[1]))))
    _, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jsonify({
        "image_id":  image_id,
        "filename":  f.filename,
        "thumbnail": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        "size":      [img.shape[1], img.shape[0]],
    })

@app.route("/api/process", methods=["POST"])
def process():
    data     = request.json or {}
    image_id = data.get("image_id")
    mode     = data.get("mode", "contour")
    params   = data.get("params", {})
    if not image_id:
        return jsonify({"error": "image_id required"}), 400
    src = None
    for ext in ALLOWED_EXTS:
        c = UPLOAD_DIR / f"{image_id}{ext}"
        if c.exists(): src = c; break
    if not src:
        return jsonify({"error": "Image not found"}), 404
    out_name = f"{image_id}_{mode}_{uuid.uuid4().hex[:6]}.svg"
    params["output"] = str(OUTPUT_DIR / out_name)
    t0 = time.time()
    try:
        svg_path = process_image(str(src), mode, params)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    svg_text = Path(svg_path).read_text(encoding="utf-8")
    n_strokes = svg_text.count("<line") + svg_text.count("<polyline")
    return jsonify({
        "svg":          svg_text,
        "svg_file":     out_name,
        "elapsed_s":    round(time.time()-t0, 2),
        "stroke_count": n_strokes,
        "mode":         mode,
    })

@app.route("/api/download/<filename>")
def download(filename):
    p = OUTPUT_DIR / filename
    if not p.exists() or p.suffix != ".svg":
        return jsonify({"error": "Not found"}), 404
    return send_file(str(p), mimetype="image/svg+xml",
                     as_attachment=True, download_name=filename)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("Plotter server → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)