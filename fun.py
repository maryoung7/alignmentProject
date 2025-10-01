import cv2
import numpy as np
import itertools

# =========================
# Config
# =========================
RESIZE_TO = (640, 480)
CLAHE_CLIP = 2.5
KEEP_LEN_FRAC = 0.03
MERGE_PARALLEL_DIST_PX = 8.0

# UI / interaction
PAD = 80
HANDLE_R = 10
PICK_R2 = 18**2
FINAL_PAD = 80
FINAL_BORDER_MODE = cv2.BORDER_REPLICATE
FINAL_BORDER_VAL = (32, 32, 32)

# Optional real-world sizing
UNITS = "mm"                 # label only
KNOWN_WIDTH_UNITS  = None    # e.g., 210.0 for A4 width in mm. Set one of these or None.
KNOWN_HEIGHT_UNITS = None    # e.g., 297.0 for A4 height in mm.

# Camera intrinsics for tilt estimate (fx=fy is ok). If None, we guess from frame size.
FX_FY_PIX = None             # e.g., 1200.0  (pixels)

# =========================
# Preprocessing
# =========================
def preprocess_image(img_bgr_or_gray, resize_to=RESIZE_TO):
    if img_bgr_or_gray.ndim == 2:
        gray = img_bgr_or_gray
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr  = img_bgr_or_gray
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if resize_to is not None:
        bgr  = cv2.resize(bgr, resize_to)
        gray = cv2.resize(gray, resize_to)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    gx = cv2.Scharr(enhanced, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(enhanced, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # percentile-based Canny thresholds (robust to lighting)
    p10, p90 = np.percentile(mag, (10, 90))
    lo = int(max(0, 0.8 * p10))
    hi = int(min(255, 1.2 * p90))
    edges = cv2.Canny(mag, lo, hi, L2gradient=True)
    return bgr, enhanced, edges

# =========================
# Detection & geometry
# =========================
def detect_lines(enhanced_gray, edges=None, use_edges=False):
    h, w = enhanced_gray.shape[:2]
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    if use_edges and edges is not None:
        lines, _, _, _ = lsd.detect(edges)
    else:
        lines, _, _, _ = lsd.detect(enhanced_gray)
    segs = np.array([l[0] for l in lines], np.float32) if lines is not None else np.empty((0,4), np.float32)
    if segs.size:
        diag = np.hypot(w, h)
        keep = [s for s in segs if np.hypot(s[2]-s[0], s[3]-s[1]) >= KEEP_LEN_FRAC*diag]
        segs = np.array(keep, np.float32) if keep else np.empty((0,4), np.float32)
    return segs

def angle_between_lines(L1, L2):
    n1 = np.array([L1[0], L1[1]])
    n2 = np.array([L2[0], L2[1]])
    cosang = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(abs(cosang), -1, 1)))  # 0 parallel, 90 perp
    return ang

def line_from_segment(seg):
    x1, y1, x2, y2 = seg
    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])
    L  = np.cross(p1, p2)   # ax+by+c=0
    n  = np.hypot(L[0], L[1])
    return L / (n + 1e-9)

def intersect(L1, L2):
    p = np.cross(L1, L2)
    if abs(p[2]) < 1e-9: return None
    return (p[:2] / p[2]).astype(np.float32)

def find_perpendicular_corners(segs, angle_tol=40, min_dist=20):
    if len(segs) == 0: return None
    lines = [line_from_segment(s) for s in segs]
    candidates = []
    for (L1, L2) in itertools.combinations(lines, 2):
        ang = angle_between_lines(L1, L2)
        if 90-angle_tol <= ang <= 90+angle_tol:
            pt = intersect(L1, L2)
            if pt is not None:
                candidates.append(pt)
    if not candidates:
        return None
    pts = np.array(candidates)
    final = []
    for p in pts:
        if all(np.linalg.norm(p - q) > min_dist for q in final):
            final.append(p)
    return np.array(final, np.float32) if final else None

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_from_points(src_img, pts, final_pad=FINAL_PAD, mode=FINAL_BORDER_MODE, value=FINAL_BORDER_VAL):
    rect = order_points(pts)
    wA = np.linalg.norm(rect[0] - rect[1])
    wB = np.linalg.norm(rect[2] - rect[3])
    hA = np.linalg.norm(rect[0] - rect[3])
    hB = np.linalg.norm(rect[1] - rect[2])
    W = max(1, int(max(wA, wB)))
    H = max(1, int(max(hA, hB)))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src_img, M, (W, H))
    cx, cy = W // 2, H // 2
    cv2.line(warped, (cx, 0), (cx, H), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (W, cy), (0, 255, 0), 1)
    if final_pad > 0:
        warped = cv2.copyMakeBorder(warped, final_pad, final_pad, final_pad, final_pad,
                                    borderType=mode, value=value)
    return warped, M, (W, H)

# ---------- New: metrics ----------
def estimate_metrics(pts, H_img_to_rect, out_size, frame_shape):
    """
    Returns dict with pixel size, aspect, area, in-plane rotation,
    and estimated tilt angle (deg) via homography decomposition.
    """
    rect = order_points(pts)
    # pixel size (average opposite edges)
    top_w    = np.linalg.norm(rect[0] - rect[1])
    bottom_w = np.linalg.norm(rect[2] - rect[3])
    left_h   = np.linalg.norm(rect[0] - rect[3])
    right_h  = np.linalg.norm(rect[1] - rect[2])
    Wpx = 0.5 * (top_w + bottom_w)
    Hpx = 0.5 * (left_h + right_h)
    aspect = Wpx / (Hpx + 1e-9)
    area = Wpx * Hpx

    # in-plane rotation: angle of top edge vs. horizontal (degrees, [-180,180))
    dy, dx = rect[1][1] - rect[0][1], rect[1][0] - rect[0][0]
    inplane_deg = np.degrees(np.arctan2(dy, dx))

    # out-of-plane tilt (estimate): need K and plane-to-image homography
    h, w = frame_shape[:2]
    fx_fy = FX_FY_PIX if FX_FY_PIX is not None else 0.9 * max(w, h)  # heuristic if unknown
    K = np.array([[fx_fy, 0, w/2.0],
                  [0, fx_fy, h/2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # We have H_img_to_rect (image -> fronto-parallel). Invert to get plane->image.
    H = np.linalg.inv(H_img_to_rect).astype(np.float64)

    A = np.linalg.inv(K) @ H
    a1 = A[:, 0]; a2 = A[:, 1]
    # Normalize with shared scale; orthonormalize via SVD
    r1 = a1 / (np.linalg.norm(a1) + 1e-12)
    r2 = a2 / (np.linalg.norm(a2) + 1e-12)
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt  # closest rotation
    n = R[:, 2] # plane normal in camera coords
    # tilt: angle between normal and camera Z axis
    tilt_deg = np.degrees(np.arccos(np.clip(abs(n[2]) / (np.linalg.norm(n) + 1e-12), -1, 1)))

    # Optional real-world size if a reference is provided
    real_W = real_H = None
    if KNOWN_WIDTH_UNITS is not None and KNOWN_WIDTH_UNITS > 0:
        scale = KNOWN_WIDTH_UNITS / (Wpx + 1e-9)
        real_W = KNOWN_WIDTH_UNITS
        real_H = Hpx * scale
    elif KNOWN_HEIGHT_UNITS is not None and KNOWN_HEIGHT_UNITS > 0:
        scale = KNOWN_HEIGHT_UNITS / (Hpx + 1e-9)
        real_H = KNOWN_HEIGHT_UNITS
        real_W = Wpx * scale

    return {
        "Wpx": Wpx, "Hpx": Hpx, "aspect": aspect, "area_px2": area,
        "inplane_deg": inplane_deg, "tilt_deg": tilt_deg,
        "real_W": real_W, "real_H": real_H
    }

def render_metrics_panel(metrics, out_size):
    W, H = out_size
    panel = np.full((220, 380, 3), 250, np.uint8)
    x, y = 10, 25
    lh = 24
    def put(line):
        nonlocal y
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        y += lh

    put("Rectangle Metrics")
    put("-----------------")
    put(f"Width (px):  {metrics['Wpx']:.1f}")
    put(f"Height (px): {metrics['Hpx']:.1f}")
    put(f"Aspect:      {metrics['aspect']:.3f}")
    put(f"Area (px^2): {metrics['area_px2']:.0f}")
    put(f"In-plane rot: {metrics['inplane_deg']:+.1f} deg")
    put(f"Tilt (est.):  {metrics['tilt_deg']:.1f} deg")
    if metrics["real_W"] is not None and metrics["real_H"] is not None:
        put(f"Size: {metrics['real_W']:.1f} x {metrics['real_H']:.1f} {UNITS}")
    else:
        put(f"(Set KNOWN_WIDTH_UNITS or KNOWN_HEIGHT_UNITS for real size)")
    put("")
    put(f"Warp out: {W} x {H} px")
    return panel

# =========================
# Guess + interactive UI
# =========================
def guess_quadrilateral(work_bgr):
    _, enhanced, edges = preprocess_image(work_bgr)
    segs = detect_lines(enhanced, edges, use_edges=False)
    if len(segs) < 4:
        segs = detect_lines(enhanced, edges, use_edges=True)
    corners = find_perpendicular_corners(segs, angle_tol=40, min_dist=20)
    if corners is None or len(corners) < 4:
        return None
    rect = cv2.minAreaRect(corners)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return [tuple(map(float, p)) for p in box]

def default_quad_for_canvas(work_shape):
    h, w = work_shape[:2]
    x0, y0 = PAD, PAD
    x1, y1 = w - PAD, h - PAD
    inset = int(0.08 * min(x1 - x0, y1 - y0))
    return [
        (x0 + inset, y0 + inset),
        (x1 - inset, y0 + inset),
        (x1 - inset, y1 - inset),
        (x0 + inset, y1 - inset),
    ]

def draw_preview(img, pts):
    vis = img.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (PAD, PAD), (w - PAD - 1, h - PAD - 1), (128, 128, 128), 1, cv2.LINE_AA)
    if len(pts) == 4:
        ordered = order_points(pts)
        cv2.polylines(vis, [ordered.astype(np.int32)], True, (0, 255, 0), 2)
        for i, p in enumerate(ordered):
            p = tuple(map(int, p))
            cv2.circle(vis, p, HANDLE_R, (0, 0, 255), -1)
            cv2.circle(vis, p, HANDLE_R, (255, 255, 255), 1)
            cv2.putText(vis, str(i), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Drag corners to adjust. 'r' re-detect, 'q' quit.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)
    return vis

def nearest_handle(x, y, pts):
    if len(pts) != 4: return (None, float("inf"))
    q = np.array([x, y], dtype=np.float32)
    d2 = [float(np.sum((q - p)**2)) for p in pts]
    i = int(np.argmin(d2))
    return (i, d2[i]) if d2[i] <= PICK_R2 else (None, float("inf"))

# =========================
# Globals for mouse
# =========================
points = []
drag_idx = None
work = None

def on_mouse(event, x, y, flags, param):
    global drag_idx, points
    if work is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        i, _ = nearest_handle(x, y, points)
        drag_idx = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            h, w = work.shape[:2]
            nx = max(0, min(w - 1, x))
            ny = max(0, min(h - 1, y))
            points[drag_idx] = (float(nx), float(ny))
    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None

# =========================
# Main
# =========================
if __name__ == "__main__":
    img = cv2.imread("img 3.png")
    if img is None:
        raise SystemExit("Could not read image")
    img = cv2.resize(img, RESIZE_TO)

    work = cv2.copyMakeBorder(
        img, PAD, PAD, PAD, PAD,
        borderType=cv2.BORDER_REPLICATE
    )

    points = guess_quadrilateral(work)
    if points is None:
        points = default_quad_for_canvas(work.shape)

    cv2.namedWindow("Input")
    cv2.setMouseCallback("Input", on_mouse)
    cv2.namedWindow("Warped")
    cv2.namedWindow("Metrics")

    while True:
        preview = draw_preview(work, points)
        cv2.imshow("Input", preview)

        if len(points) == 4:
            warped, H, out_size = warp_from_points(work, points)
            cv2.imshow("Warped", warped)

            # ---- New: compute & display metrics ----
            m = estimate_metrics(points, H, out_size, work.shape)
            panel = render_metrics_panel(m, out_size)
            cv2.imshow("Metrics", panel)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            points = guess_quadrilateral(work)
            if points is None:
                points = default_quad_for_canvas(work.shape)

    cv2.destroyAllWindows()
