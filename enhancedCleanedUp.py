import cv2
import numpy as np
import itertools
# =========================
# Config
# =========================
RESIZE_TO = (640, 480)
CLAHE_CLIP = 2.5
KEEP_LEN_FRAC = 0.03
KMEANS_ANGLE_SEP_DEG = 25
MERGE_PARALLEL_DIST_PX = 8.0

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

    v   = np.median(mag)
    lo  = int(max(0, 0.66*v))
    hi  = int(min(255, 1.33*v))
    edges = cv2.Canny(mag, lo, hi, L2gradient=True)

    return bgr, enhanced, edges

# =========================
# Line detection
# =========================
def detect_lines(enhanced_gray, edges=None, use_edges=False):
    h, w = enhanced_gray.shape[:2]
    base_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

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

    overlay = base_bgr.copy()
    for x1,y1,x2,y2 in segs.astype(int):
        cv2.line(overlay, (x1,y1), (x2,y2), (0,255,255), 1, cv2.LINE_AA)

    return segs, overlay

# =========================
# Geometry helpers
# =========================
def angle_between_lines(L1, L2):
    # L = (a, b, c), normal vector n = (a,b)
    n1 = np.array([L1[0], L1[1]])
    n2 = np.array([L2[0], L2[1]])
    cosang = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(abs(cosang), -1, 1)))  # 0 = parallel, 90 = perp
    return ang

def line_from_segment(seg):
    x1,y1,x2,y2 = seg
    p1 = np.array([x1,y1,1.0])
    p2 = np.array([x2,y2,1.0])
    L  = np.cross(p1, p2)   # ax+by+c=0
    n  = np.hypot(L[0], L[1])
    return L / (n + 1e-9)

def intersect(L1, L2):
    p = np.cross(L1, L2)
    if abs(p[2]) < 1e-9: return None
    return (p[:2] / p[2]).astype(np.float32)

def find_perpendicular_corners(segs, angle_tol=10, min_dist=20):
    """Return candidate corners where two lines intersect ~90°."""
    lines = [line_from_segment(s) for s in segs]
    candidates = []
    for (L1,L2) in itertools.combinations(lines, 2):
        ang = angle_between_lines(L1, L2)
        if 90-angle_tol <= ang <= 90+angle_tol:
            pt = intersect(L1, L2)
            if pt is not None:
                candidates.append(pt)
    if not candidates:
        return None
    # cluster / deduplicate
    pts = np.array(candidates)
    final = []
    for p in pts:
        if all(np.linalg.norm(p - q) > min_dist for q in final):
            final.append(p)
    return np.array(final, np.float32)
def warp_from_corners(img, corners):
    """
    Warp image using 4 corners (tl,tr,br,bl).
    """
    rect = np.array(corners, np.float32)

    # compute target width & height from distances
    wA = np.linalg.norm(rect[0] - rect[1])
    wB = np.linalg.norm(rect[2] - rect[3])
    hA = np.linalg.norm(rect[0] - rect[3])
    hB = np.linalg.norm(rect[1] - rect[2])
    W = int(max(wA, wB))
    H = int(max(hA, hB))

    dst = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]], np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (W, H))

    # draw crosshair for alignment
    cx, cy = W // 2, H // 2
    cv2.line(warped, (cx, 0), (cx, H), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (W, cy), (0, 255, 0), 1)

    return warped

# =========================
# Main demo
# =========================
if __name__ == "__main__":
    img = cv2.imread("img 3.png")
    img = cv2.resize(img, (640, 480))
    if img is None:
        raise SystemExit("Could not read image")

    bgr, enhanced, edges = preprocess_image(img)

    segs, overlay = detect_lines(enhanced, edges, use_edges=False)
    if len(segs) < 4:
        segs, overlay = detect_lines(enhanced, edges, use_edges=True)
    corners = find_perpendicular_corners(segs, angle_tol=40, min_dist=20)
    print("Detected corners:", corners)
    vis = img.copy()
    if corners is not None and len(corners) >= 4:
        # fit a minimum-area rectangle to candidate corners
        rect = cv2.minAreaRect(corners)          # center, (w,h), angle
        box  = cv2.boxPoints(rect)               # 4 corner points
        box  = np.array(box, dtype=np.float32)

        # draw detected quad
        for (x,y) in box.astype(int):
            cv2.circle(vis, (x,y), 5, (0,0,255), -1)
        cv2.polylines(vis, [box.astype(int)], True, (0,255,0), 2)

        # warp to top-down view
        warped = warp_from_corners(img, box)
        cv2.imshow("Warped", warped)

    cv2.imshow("Corners", vis)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
