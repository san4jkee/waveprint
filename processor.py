"""
Plotter Image Processor
Converts raster images into SVG line art suitable for pen plotters.
Modes: contour, hatching, stippling, flowfield
"""

import cv2
import numpy as np
import svgwrite
import math
import random
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path: str, max_dim: int = 1200) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def ramer_douglas_peucker(pts, epsilon=1.0):
    """Simplify a polyline using RDP algorithm."""
    if len(pts) < 3:
        return pts
    start, end = np.array(pts[0]), np.array(pts[-1])
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        dists = [np.linalg.norm(np.array(p) - start) for p in pts]
    else:
        dists = [abs(np.cross(line, np.array(p) - start)) / norm for p in pts]
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = ramer_douglas_peucker(pts[:idx+1], epsilon)
        right = ramer_douglas_peucker(pts[idx:], epsilon)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def sorted_strokes(strokes):
    """Greedy nearest-neighbour sort to minimise pen travel."""
    if not strokes:
        return strokes
    result = [strokes[0]]
    remaining = list(strokes[1:])
    cur_end = strokes[0][-1]
    while remaining:
        best_i, best_d, best_rev = 0, float('inf'), False
        for i, s in enumerate(remaining):
            d0 = math.dist(cur_end, s[0])
            d1 = math.dist(cur_end, s[-1])
            d = min(d0, d1)
            if d < best_d:
                best_d, best_i, best_rev = d, i, (d1 < d0)
        s = remaining.pop(best_i)
        if best_rev:
            s = list(reversed(s))
        result.append(s)
        cur_end = result[-1][-1]
    return result


def save_svg(strokes, w, h, stroke_width=0.5, output_path="out.svg"):
    dwg = svgwrite.Drawing(output_path,
                           size=(f"{w}px", f"{h}px"),
                           viewBox=f"0 0 {w} {h}")
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))
    g = dwg.add(dwg.g(stroke="black", fill="none",
                      stroke_width=stroke_width, stroke_linecap="round"))
    for pts in strokes:
        # ensure plain Python floats (not numpy scalars) for svgwrite
        clean = [(round(float(x), 3), round(float(y), 3)) for x, y in pts]
        if len(clean) == 2:
            g.add(dwg.line(start=clean[0], end=clean[1]))
        elif len(clean) > 2:
            g.add(dwg.polyline(clean))
    dwg.save()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Contour tracing
# ─────────────────────────────────────────────────────────────────────────────

def process_contour(gray, params):
    """
    Edge detection via Canny + contour tracing.
    params:
      blur_radius  – gaussian blur kernel (odd int, 3–15)
      canny_low    – lower Canny threshold (10–100)
      canny_high   – upper Canny threshold (50–300)
      simplify     – RDP epsilon for polyline simplification (0–5)
      min_length   – discard contours shorter than this (px)
      stroke_width – SVG stroke width
    """
    blur_r    = int(params.get("blur_radius", 5)) | 1       # ensure odd
    c_low     = int(params.get("canny_low", 40))
    c_high    = int(params.get("canny_high", 120))
    simplify  = float(params.get("simplify", 1.5))
    min_len   = int(params.get("min_length", 10))
    sw        = float(params.get("stroke_width", 0.7))

    blurred = cv2.GaussianBlur(gray, (blur_r, blur_r), 0)
    edges   = cv2.Canny(blurred, c_low, c_high)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    strokes = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).tolist()
        if len(pts) < min_len:
            continue
        pts = [(float(p[0]), float(p[1])) for p in pts]
        if simplify > 0:
            pts = ramer_douglas_peucker(pts, simplify)
        if len(pts) >= 2:
            strokes.append(pts)

    strokes = sorted_strokes(strokes)
    h, w = gray.shape
    return save_svg(strokes, w, h, sw,
                    params.get("output", "out_contour.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Hatching
# ─────────────────────────────────────────────────────────────────────────────

def process_hatching(gray, params):
    """
    Parallel line hatching. Brightness → gap between strokes.
    params:
      angle        – hatch angle in degrees (0=horizontal, 45=diagonal)
      line_spacing – pixels between hatch rows (2–20)
      cross_hatch  – bool, add a second layer at 90° offset
      stroke_width – SVG stroke width
      invert       – bool, dark lines on white vs light on dark
      contour_mix  – bool, also draw contour edges on top
    """
    angle        = float(params.get("angle", 45))
    spacing      = float(params.get("line_spacing", 6))
    cross        = bool(params.get("cross_hatch", False))
    sw           = float(params.get("stroke_width", 0.5))
    invert       = bool(params.get("invert", False))
    contour_mix  = bool(params.get("contour_mix", True))

    h, w = gray.shape

    def hatch_layer(img, ang):
        rad = math.radians(ang)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # rotate image so we always scan horizontal rows
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        rh = int(abs(h * cos_a) + abs(w * sin_a))
        rw = int(abs(w * cos_a) + abs(h * sin_a))
        rotated = cv2.warpAffine(img, M, (rw, rh),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=255)
        Mi = cv2.invertAffineTransform(M)

        strokes = []
        y = 0
        while y < rh:
            row = rotated[int(y)] if int(y) < rh else None
            if row is None:
                break
            x = 0
            while x < rw:
                px = row[int(x)] if int(x) < rw else 255
                bright = px / 255.0
                val = bright if not invert else (1.0 - bright)
                # stroke_len grows with darkness
                stroke_len = max(0, (1.0 - val) * spacing * 1.5)
                gap_len    = max(0.5, val * spacing * 1.5 + 1)
                if stroke_len > 0.5:
                    # transform back to original coords
                    p1r = np.array([[[x,            float(y)]]], dtype=np.float32)
                    p2r = np.array([[[x+stroke_len, float(y)]]], dtype=np.float32)
                    p1 = cv2.transform(p1r, Mi)[0][0]
                    p2 = cv2.transform(p2r, Mi)[0][0]
                    if (0 <= p1[0] < w and 0 <= p1[1] < h and
                            0 <= p2[0] < w and 0 <= p2[1] < h):
                        strokes.append([(float(p1[0]), float(p1[1])),
                                        (float(p2[0]), float(p2[1]))])
                    x += stroke_len + gap_len
                else:
                    x += gap_len
            y += spacing
        return strokes

    strokes = hatch_layer(gray, angle)
    if cross:
        strokes += hatch_layer(gray, angle + 90)

    if contour_mix:
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blur, 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_TC89_L1)
        for cnt in cnts:
            pts = cnt.reshape(-1, 2).tolist()
            if len(pts) < 8:
                continue
            pts = [(float(p[0]), float(p[1])) for p in pts]
            strokes.append(ramer_douglas_peucker(pts, 1.0))

    strokes = sorted_strokes(strokes)
    return save_svg(strokes, w, h, sw,
                    params.get("output", "out_hatching.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: Stippling (rejection sampling + short strokes)
# ─────────────────────────────────────────────────────────────────────────────

def process_stippling(gray, params):
    """
    Dark-region weighted point placement, rendered as tiny strokes.
    params:
      n_points   – number of stipple strokes (500–8000)
      dot_len    – length of each mini-stroke in px (0=circle dot)
      blur_radius – pre-blur to smooth probability map
      stroke_width
    """
    n_pts    = int(params.get("n_points", 3000))
    dot_len  = float(params.get("dot_len", 2.0))
    blur_r   = int(params.get("blur_radius", 9)) | 1
    sw       = float(params.get("stroke_width", 0.6))

    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (blur_r, blur_r), 0)
    # probability = darkness
    prob = 1.0 - blurred.astype(np.float64) / 255.0
    prob = np.clip(prob, 0, None)
    total = prob.sum()
    if total == 0:
        prob[:] = 1.0
        total = prob.sum()
    prob_flat = (prob / total).ravel()

    np.random.seed(42)
    indices = np.random.choice(len(prob_flat), size=n_pts, p=prob_flat)
    ys, xs = np.unravel_index(indices, (h, w))

    strokes = []
    for x, y in zip(xs, ys):
        # random angle per stroke for organic look
        angle = random.uniform(0, math.pi)
        dx = math.cos(angle) * dot_len / 2
        dy = math.sin(angle) * dot_len / 2
        x1 = max(0, min(w-1, x - dx))
        y1 = max(0, min(h-1, y - dy))
        x2 = max(0, min(w-1, x + dx))
        y2 = max(0, min(h-1, y + dy))
        strokes.append([(float(x1), float(y1)), (float(x2), float(y2))])

    strokes = sorted_strokes(strokes)
    return save_svg(strokes, w, h, sw,
                    params.get("output", "out_stippling.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4: Flow field (gradient-directed curves)
# ─────────────────────────────────────────────────────────────────────────────

def process_flowfield(gray, params):
    """
    Particles follow image gradient direction, producing organic curves.
    params:
      n_lines      – number of flow lines (200–2000)
      step_size    – particle step in px (0.5–5)
      max_steps    – max steps per particle (10–200)
      blur_radius  – blur before gradient (larger = smoother flow)
      min_steps    – discard lines shorter than this
      stroke_width
      perpendicular – bool, follow gradient perpendicular (default True)
    """
    n_lines   = int(params.get("n_lines", 600))
    step      = float(params.get("step_size", 2.0))
    max_steps = int(params.get("max_steps", 80))
    min_steps = int(params.get("min_steps", 4))
    blur_r    = int(params.get("blur_radius", 11)) | 1
    sw        = float(params.get("stroke_width", 0.4))
    perp      = bool(params.get("perpendicular", True))

    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (blur_r, blur_r), 0).astype(np.float32)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
    angles = np.arctan2(gy, gx)   # angle of gradient
    mag    = np.sqrt(gx**2 + gy**2)
    mag_norm = mag / (mag.max() + 1e-6)

    strokes = []
    np.random.seed(7)
    starts_x = np.random.rand(n_lines) * w
    starts_y = np.random.rand(n_lines) * h

    for sx, sy in zip(starts_x, starts_y):
        x, y = float(sx), float(sy)
        pts = [(x, y)]
        for _ in range(max_steps):
            xi = int(np.clip(x, 0, w-1))
            yi = int(np.clip(y, 0, h-1))
            a = angles[yi, xi]
            if perp:
                a += math.pi / 2          # perpendicular to gradient = follow isocurves
            # vary step by local gradient magnitude for denser dark zones
            local_step = step * (0.5 + mag_norm[yi, xi] * 0.5)
            nx = x + math.cos(a) * local_step
            ny = y + math.sin(a) * local_step
            if not (0 <= nx < w and 0 <= ny < h):
                break
            x, y = nx, ny
            pts.append((x, y))

        if len(pts) >= min_steps:
            strokes.append(pts)

    strokes = sorted_strokes(strokes)
    return save_svg(strokes, w, h, sw,
                    params.get("output", "out_flowfield.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "contour":   process_contour,
    "hatching":  process_hatching,
    "stippling": process_stippling,
    "flowfield": process_flowfield,
}


def process_image(image_path: str, mode: str, params: dict) -> str:
    """
    Main entry point.
    Returns path to generated SVG file.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODES)}")
    gray = load_gray(image_path, max_dim=int(params.get("max_dim", 1000)))
    return MODES[mode](gray, params)


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Plotter image processor")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("mode", choices=list(MODES),
                        help="Processing mode")
    parser.add_argument("--params", default="{}", type=json.loads,
                        help='JSON params, e.g. \'{"line_spacing":8}\'')
    parser.add_argument("--output", default=None,
                        help="Output SVG path (overrides params.output)")
    args = parser.parse_args()

    p = args.params
    if args.output:
        p["output"] = args.output

    out = process_image(args.image, args.mode, p)
    print(f"Saved: {out}")
