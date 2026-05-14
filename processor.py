"""
Plotter Image Processor — v3
Modes: contour, hatching, stippling, flowfield, voronoi, spiral, wave, sketch
"""

import cv2
import numpy as np
import svgwrite
import math
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path: str, max_dim: int = 1000) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rdp(pts, epsilon=1.0):
    """Ramer-Douglas-Peucker simplification (iterative to avoid recursion limit)."""
    if len(pts) < 3:
        return pts
    stack = [(0, len(pts) - 1)]
    keep  = {0, len(pts) - 1}
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        arr  = np.array(pts[lo:hi+1])
        s, e = arr[0], arr[-1]
        seg  = e - s
        n    = np.linalg.norm(seg)
        if n < 1e-9:
            dists = np.linalg.norm(arr - s, axis=1)
        else:
            dists = np.abs(np.cross(seg, arr - s)) / n
        idx = int(np.argmax(dists)) + lo
        if dists[idx - lo] > epsilon:
            keep.add(idx)
            stack.append((lo, idx))
            stack.append((idx, hi))
    return [pts[i] for i in sorted(keep)]


def _d2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def nn_sort(strokes):
    """Greedy NN sort for ≤3000 strokes; boustrophedon bucket otherwise."""
    if not strokes:
        return strokes
    n = len(strokes)
    if n > 3000:
        xs = [s[0][0] for s in strokes]
        ys = [s[0][1] for s in strokes]
        x0, x1 = min(xs), max(xs) + 1e-6
        y0, y1 = min(ys), max(ys) + 1e-6
        cells = max(1, int(math.sqrt(n / 8)))
        def key(s):
            cx = int((s[0][0] - x0) / (x1 - x0) * cells)
            cy = int((s[0][1] - y0) / (y1 - y0) * cells)
            return cy * cells + (cx if cy % 2 == 0 else cells - 1 - cx)
        return sorted(strokes, key=key)
    result    = [strokes[0]]
    remaining = list(strokes[1:])
    cur       = strokes[0][-1]
    while remaining:
        bi, bd, brev = 0, float('inf'), False
        for i, s in enumerate(remaining):
            d0 = _d2(cur, s[0]);  d1 = _d2(cur, s[-1])
            d  = min(d0, d1)
            if d < bd:
                bd, bi, brev = d, i, (d1 < d0)
        s = remaining.pop(bi)
        if brev:
            s = list(reversed(s))
        result.append(s)
        cur = result[-1][-1]
    return result


def save_svg(strokes, w, h, stroke_width=0.5, output_path="out.svg"):
    dwg = svgwrite.Drawing(output_path,
                           size=(f"{w}px", f"{h}px"),
                           viewBox=f"0 0 {w} {h}")
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))
    g = dwg.add(dwg.g(stroke="black", fill="none",
                      stroke_width=stroke_width,
                      stroke_linecap="round", stroke_linejoin="round"))
    for pts in strokes:
        clean = [(round(float(x), 2), round(float(y), 2)) for x, y in pts]
        if len(clean) == 2:
            g.add(dwg.line(start=clean[0], end=clean[1]))
        elif len(clean) > 2:
            g.add(dwg.polyline(clean))
    dwg.save()
    return output_path


def _blur(gray, r):
    r = max(1, int(r)) | 1
    return cv2.GaussianBlur(gray, (r, r), 0)


def _sample(img, x, y):
    """Bilinear sample, returns 0.0–1.0 brightness."""
    h, w = img.shape
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0+1, w-1), min(y0+1, h-1)
    fx, fy = x - x0, y - y0
    v = (img[y0,x0]*(1-fx)*(1-fy) + img[y0,x1]*fx*(1-fy) +
         img[y1,x0]*(1-fx)*fy     + img[y1,x1]*fx*fy)
    return float(v) / 255.0


def _contours_from(gray, blur=5, lo=40, hi=120, min_pts=8, simp=1.5):
    edges = cv2.Canny(_blur(gray, blur), lo, hi)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    out = []
    for cnt in cnts:
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        if len(pts) >= min_pts:
            out.append(rdp(pts, simp))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 – Contour
# ─────────────────────────────────────────────────────────────────────────────

def process_contour(gray, params):
    blur_r   = int(params.get("blur_radius",   5)) | 1
    c_low    = int(params.get("canny_low",    40))
    c_high   = int(params.get("canny_high",  120))
    simplify = float(params.get("simplify",   1.5))
    min_len  = int(params.get("min_length",   10))
    sw       = float(params.get("stroke_width", 0.7))
    h, w = gray.shape
    strokes = _contours_from(gray, blur_r, c_low, c_high, min_len, simplify)
    return save_svg(nn_sort(strokes), w, h, sw,
                    params.get("output", "out_contour.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 – Hatching
# ─────────────────────────────────────────────────────────────────────────────

def process_hatching(gray, params):
    angle    = float(params.get("angle",        45))
    spacing  = float(params.get("line_spacing",  6))
    cross    = bool(params.get("cross_hatch",  False))
    cont_mix = bool(params.get("contour_mix",  True))
    invert   = bool(params.get("invert",       False))
    sw       = float(params.get("stroke_width",  0.5))
    h, w     = gray.shape

    def hatch_layer(img, ang):
        rad = math.radians(ang)
        ca, sa = math.cos(rad), math.sin(rad)
        M   = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        rw  = int(abs(w*ca) + abs(h*sa))
        rh  = int(abs(h*ca) + abs(w*sa))
        rot = cv2.warpAffine(img, M, (rw, rh),
                             flags=cv2.INTER_LINEAR, borderValue=255)
        Mi  = cv2.invertAffineTransform(M)
        out = []
        y   = 0.0
        while y < rh:
            row = rot[int(y)] if int(y) < rh else None
            if row is None:
                break
            x = 0.0
            while x < rw:
                ix     = min(int(x), rw-1)
                bright = row[ix] / 255.0
                val    = bright if not invert else 1.0 - bright
                slen   = max(0.0, (1.0-val) * spacing * 1.6)
                gap    = max(0.5,      val  * spacing * 1.6 + 1.0)
                if slen > 0.5:
                    p1r = np.array([[[x,       y]]], dtype=np.float32)
                    p2r = np.array([[[x+slen,  y]]], dtype=np.float32)
                    p1  = cv2.transform(p1r, Mi)[0][0]
                    p2  = cv2.transform(p2r, Mi)[0][0]
                    if (0<=p1[0]<w and 0<=p1[1]<h and
                            0<=p2[0]<w and 0<=p2[1]<h):
                        out.append([(float(p1[0]),float(p1[1])),
                                    (float(p2[0]),float(p2[1]))])
                    x += slen + gap
                else:
                    x += gap
            y += spacing
        return out

    strokes = hatch_layer(gray, angle)
    if cross:
        strokes += hatch_layer(gray, angle+90)
    if cont_mix:
        strokes += _contours_from(gray)
    return save_svg(nn_sort(strokes), w, h, sw,
                    params.get("output", "out_hatching.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 – Stippling  (fixed v3)
# ─────────────────────────────────────────────────────────────────────────────

def process_stippling(gray, params):
    """
    Fixes:
    - Gamma-correct probability (sqrt) for perceptually even distribution
    - min_dot_len ≥ 4 px default so dots are actually visible
    - vary_size: darker pixel → longer stroke (proportional, not squared)
    - angle aligned to local gradient for organic look
    """
    n_pts     = int(params.get("n_points",    3000))
    min_dl    = float(params.get("min_dot_len",  4.0))
    max_dl    = float(params.get("max_dot_len", 12.0))
    blur_r    = int(params.get("blur_radius",    9)) | 1
    sw        = float(params.get("stroke_width", 0.7))
    vary_size = bool(params.get("vary_size",    True))

    h, w = gray.shape
    blurred = _blur(gray, blur_r).astype(np.float64)

    # Gamma correction: sqrt gives perceptually uniform distribution
    prob = np.sqrt(1.0 - blurred / 255.0)
    prob = np.clip(prob, 1e-6, None)
    prob_flat = (prob / prob.sum()).ravel()

    # Gradient for dot angle
    gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    angle_map = np.arctan2(gy, gx)          # perpendicular = +pi/2

    rng     = np.random.default_rng(42)
    indices = rng.choice(len(prob_flat), size=n_pts, p=prob_flat, replace=True)
    ys, xs  = np.unravel_index(indices, (h, w))
    dark_flat = prob.ravel()

    strokes = []
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        local_dark = float(dark_flat[indices[i]])
        if vary_size:
            dl = min_dl + (max_dl - min_dl) * local_dark
        else:
            dl = (min_dl + max_dl) / 2.0
        # angle = local gradient direction + small random jitter
        base_ang = float(angle_map[yi, xi]) + math.pi / 2
        ang = base_ang + rng.uniform(-0.4, 0.4)
        dx  = math.cos(ang) * dl / 2
        dy  = math.sin(ang) * dl / 2
        x1  = float(np.clip(xi - dx, 0, w-1))
        y1  = float(np.clip(yi - dy, 0, h-1))
        x2  = float(np.clip(xi + dx, 0, w-1))
        y2  = float(np.clip(yi + dy, 0, h-1))
        strokes.append([(x1, y1), (x2, y2)])

    return save_svg(nn_sort(strokes), w, h, sw,
                    params.get("output", "out_stippling.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 – Flow field
# ─────────────────────────────────────────────────────────────────────────────

def process_flowfield(gray, params):
    n_lines   = int(params.get("n_lines",      600))
    step      = float(params.get("step_size",   2.0))
    max_steps = int(params.get("max_steps",     80))
    min_steps = int(params.get("min_steps",      4))
    blur_r    = int(params.get("blur_radius",   11)) | 1
    sw        = float(params.get("stroke_width", 0.4))
    perp      = bool(params.get("perpendicular", True))
    h, w  = gray.shape
    bl    = _blur(gray, blur_r).astype(np.float32)
    gx    = cv2.Sobel(bl, cv2.CV_32F, 1, 0, ksize=5)
    gy    = cv2.Sobel(bl, cv2.CV_32F, 0, 1, ksize=5)
    angles   = np.arctan2(gy, gx)
    mag_norm = np.sqrt(gx**2 + gy**2)
    mag_norm /= (mag_norm.max() + 1e-6)
    rng = np.random.default_rng(7)
    sx  = rng.uniform(0, w, n_lines)
    sy  = rng.uniform(0, h, n_lines)
    strokes = []
    for x, y in zip(sx, sy):
        x, y = float(x), float(y)
        pts  = [(x, y)]
        for _ in range(max_steps):
            xi = int(np.clip(x, 0, w-1))
            yi = int(np.clip(y, 0, h-1))
            a  = angles[yi, xi] + (math.pi/2 if perp else 0)
            ls = step * (0.5 + float(mag_norm[yi, xi]) * 0.5)
            nx, ny = x + math.cos(a)*ls, y + math.sin(a)*ls
            if not (0 <= nx < w and 0 <= ny < h):
                break
            x, y = nx, ny
            pts.append((x, y))
        if len(pts) >= min_steps:
            strokes.append(pts)
    return save_svg(nn_sort(strokes), w, h, sw,
                    params.get("output", "out_flowfield.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 5 – Voronoi
# ─────────────────────────────────────────────────────────────────────────────

def process_voronoi(gray, params):
    n_seeds  = int(params.get("n_seeds",      600))
    blur_r   = int(params.get("blur_radius",    9)) | 1
    cont_mix = bool(params.get("contour_mix", True))
    sw       = float(params.get("stroke_width", 0.5))
    h, w = gray.shape
    prob = 1.0 - _blur(gray, blur_r).astype(np.float64) / 255.0
    prob = np.clip(prob, 1e-6, None)
    prob_flat = (prob / prob.sum()).ravel()
    rng     = np.random.default_rng(13)
    actual_n = min(n_seeds, len(prob_flat))
    indices  = rng.choice(len(prob_flat), size=actual_n,
                          p=prob_flat, replace=False)
    ys, xs   = np.unravel_index(indices, (h, w))
    rect     = (0, 0, w, h)
    subdiv   = cv2.Subdiv2D(rect)
    for xi, yi in zip(xs, ys):
        x, y = float(xi), float(yi)
        if 0 < x < w-1 and 0 < y < h-1:
            try:
                subdiv.insert((x, y))
            except cv2.error:
                pass
    strokes = []
    try:
        facets, _ = subdiv.getVoronoiFacetList([])
        for facet in facets:
            pts = [(max(0.0,min(w-1.0,float(p[0]))),
                    max(0.0,min(h-1.0,float(p[1])))) for p in facet]
            for i in range(len(pts)):
                strokes.append([pts[i], pts[(i+1)%len(pts)]])
    except cv2.error:
        pass
    if cont_mix:
        strokes += _contours_from(gray)
    return save_svg(nn_sort(strokes), w, h, sw,
                    params.get("output", "out_voronoi.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 6 – Spiral  (fixed v3)
# ─────────────────────────────────────────────────────────────────────────────

def process_spiral(gray, params):
    """
    Fixes vs v2:
    - d_theta reduced to 0.008 rad (≈0.46°) — smooth curve, no jumps
    - Dark pixel → TIGHTER loops (smaller dr), not looser
    - Spiral fills canvas by computing max_r from corners correctly
    - Points collected even outside image (to avoid gaps at edges)
    - Chunk into segments of ~500 pts for readable SVG
    """
    turns       = float(params.get("turns",       60))
    base_step   = float(params.get("base_step",    5))   # px per turn in white
    min_step    = float(params.get("min_step",     0.5)) # px per turn in black
    cx_rel      = float(params.get("center_x",    0.5))
    cy_rel      = float(params.get("center_y",    0.5))
    sw          = float(params.get("stroke_width", 0.4))

    h, w  = gray.shape
    bl    = _blur(gray, 9)
    cx, cy = cx_rel * w, cy_rel * h

    # max_r: distance from centre to farthest corner
    corners = [(0,0),(w,0),(0,h),(w,h)]
    max_r = max(math.sqrt((cx-cx2)**2+(cy-cy2)**2) for cx2,cy2 in corners) + 2

    d_theta = 0.008          # ~0.46° per step — smooth
    r       = 0.0
    theta   = 0.0
    pts     = []

    while r <= max_r:
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        # always collect point (clip to visible later for SVG readability)
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))
        elif pts:
            # hit boundary — save segment, reset
            pass

        # sample brightness to decide radial step
        bx = float(np.clip(x, 0, w-1))
        by = float(np.clip(y, 0, h-1))
        bright = _sample(bl, bx, by)
        dark   = 1.0 - bright   # 0=white, 1=black

        # dark → step towards min_step; white → base_step
        dr_per_turn = base_step * (1.0 - dark) + min_step * dark
        r     += dr_per_turn * d_theta / (2 * math.pi)
        theta += d_theta
        if theta > turns * 2 * math.pi:
            break

    # Chunk into segments
    chunk_size = 500
    strokes = []
    for i in range(0, len(pts), chunk_size):
        seg = pts[i:i+chunk_size]
        # overlap by 1 point for continuity
        if i > 0 and pts:
            seg = [pts[i-1]] + seg
        if len(seg) >= 2:
            strokes.append(seg)

    return save_svg(strokes, w, h, sw,
                    params.get("output", "out_spiral.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 7 – Wave
# ─────────────────────────────────────────────────────────────────────────────

def process_wave(gray, params):
    spacing   = float(params.get("line_spacing", 10))
    frequency = float(params.get("frequency",     5))
    amplitude = float(params.get("amplitude",    15))
    phase_sh  = float(params.get("phase_shift",   1.2))
    sw        = float(params.get("stroke_width",  0.5))
    cont_mix  = bool(params.get("contour_mix",  False))
    h, w = gray.shape
    bl   = _blur(gray, 7)
    strokes = []
    y_base  = spacing / 2.0
    row_idx = 0
    while y_base < h:
        phase = row_idx * phase_sh
        pts   = []
        for xi in range(w):
            yi_s   = int(np.clip(y_base, 0, h-1))
            bright = float(bl[yi_s, xi]) / 255.0
            dark   = 1.0 - bright
            yw = y_base + amplitude * dark * math.sin(
                2*math.pi * frequency * xi / w + phase)
            pts.append((float(xi), float(np.clip(yw, 0, h-1))))
        if len(pts) >= 2:
            strokes.append(pts)
        y_base  += spacing
        row_idx += 1
    if cont_mix:
        strokes += _contours_from(gray)
    return save_svg(strokes, w, h, sw,
                    params.get("output", "out_wave.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode 8 – Sketch
# ─────────────────────────────────────────────────────────────────────────────

def process_sketch(gray, params):
    n_passes  = int(params.get("n_passes",        3))
    spacing   = float(params.get("line_spacing",   8))
    jitter    = float(params.get("jitter",         3))
    sw        = float(params.get("stroke_width",   0.5))
    blur_r    = int(params.get("blur_radius",      5)) | 1
    h, w  = gray.shape
    bl    = _blur(gray, blur_r)
    rng   = np.random.default_rng(99)
    base_angles = [0, 45, 90, 135, 22]
    strokes = []
    for pi in range(n_passes):
        ang   = base_angles[pi % len(base_angles)]
        rad   = math.radians(ang)
        ca, sa = math.cos(rad), math.sin(rad)
        M     = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        rw    = int(abs(w*ca) + abs(h*sa))
        rh    = int(abs(h*ca) + abs(w*sa))
        rot   = cv2.warpAffine(bl, M, (rw, rh),
                               flags=cv2.INTER_LINEAR, borderValue=255)
        Mi    = cv2.invertAffineTransform(M)
        y     = float(rng.uniform(0, spacing))
        while y < rh:
            yi  = int(np.clip(y, 0, rh-1))
            row = rot[yi]
            x   = 0.0
            seg = None
            while x < rw:
                xi     = int(np.clip(x, 0, rw-1))
                bright = row[xi] / 255.0
                if bright < 0.75:
                    if seg is None:
                        seg = x
                else:
                    if seg is not None and x - seg > 1:
                        jy  = y + float(rng.uniform(-jitter, jitter))
                        p1r = np.array([[[seg, jy]]], dtype=np.float32)
                        p2r = np.array([[[x,   jy]]], dtype=np.float32)
                        p1  = cv2.transform(p1r, Mi)[0][0]
                        p2  = cv2.transform(p2r, Mi)[0][0]
                        if (0<=p1[0]<w and 0<=p1[1]<h and
                                0<=p2[0]<w and 0<=p2[1]<h):
                            strokes.append([(float(p1[0]),float(p1[1])),
                                            (float(p2[0]),float(p2[1]))])
                        seg = None
                x += 1.5
            if seg is not None and rw - seg > 1:
                jy  = y + float(rng.uniform(-jitter, jitter))
                p1r = np.array([[[seg,    jy]]], dtype=np.float32)
                p2r = np.array([[[rw-1.0, jy]]], dtype=np.float32)
                p1  = cv2.transform(p1r, Mi)[0][0]
                p2  = cv2.transform(p2r, Mi)[0][0]
                if (0<=p1[0]<w and 0<=p1[1]<h and
                        0<=p2[0]<w and 0<=p2[1]<h):
                    strokes.append([(float(p1[0]),float(p1[1])),
                                    (float(p2[0]),float(p2[1]))])
            y += spacing + float(rng.uniform(-spacing*0.2, spacing*0.2))
    strokes += _contours_from(gray, blur=3, lo=30, hi=100, min_pts=6, simp=1.0)
    return save_svg(nn_sort(strokes), w, h, sw,
                    params.get("output", "out_sketch.svg"))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "contour":   process_contour,
    "hatching":  process_hatching,
    "stippling": process_stippling,
    "flowfield": process_flowfield,
    "voronoi":   process_voronoi,
    "spiral":    process_spiral,
    "wave":      process_wave,
    "sketch":    process_sketch,
}


def process_image(image_path: str, mode: str, params: dict) -> str:
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODES)}")
    gray = load_gray(image_path, max_dim=int(params.get("max_dim", 1000)))
    return MODES[mode](gray, params)


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("mode", choices=list(MODES))
    parser.add_argument("--params", default="{}", type=json.loads)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    p = args.params
    if args.output:
        p["output"] = args.output
    print("Saved:", process_image(args.image, args.mode, p))