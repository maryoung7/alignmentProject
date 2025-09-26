import cv2 
import numpy as np
import itertools

# =========================
# Config
# =========================
RESIZE_TO = (640, 480)
CLAHE_CLIP = 2.5
KEEP_LEN_FRAC = 0.02
EXPECTED_SIZE = (500, 400)   # <-- expected width, height of your rectangle in pixels
SIZE_TOL = 0.8              # +/-30% tolerance

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
        keep = [s for s in segs if np.hypot(s[2]-s[0], s[3]-s[1]) >= 0.03*diag]
        segs = np.array(keep, np.float32) if keep else np.empty((0,4), np.float32)

    overlay = base_bgr.copy()
    for x1,y1,x2,y2 in segs.astype(int):
        cv2.line(overlay, (x1,y1), (x2,y2), (0,255,255), 1, cv2.LINE_AA)

    return segs, overlay

# =========================
# Geometry helpers
# =========================
def angle_between_lines(L1, L2):
    n1 = np.array([L1[0], L1[1]])
    n2 = np.array([L2[0], L2[1]])
    cosang = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(abs(cosang), -1, 1)))
    return ang

def line_from_segment(seg):
    x1,y1,x2,y2 = seg
    p1 = np.array([x1,y1,1.0])
    p2 = np.array([x2,y2,1.0])
    L  = np.cross(p1, p2)
    n  = np.hypot(L[0], L[1])
    return L / (n + 1e-9)

def intersect(L1, L2):
    p = np.cross(L1, L2)
    if abs(p[2]) < 1e-9: return None
    return (p[:2] / p[2]).astype(np.float32)

# --- extra helpers you don't have yet ---
def seg_angle(seg):
    x1,y1,x2,y2 = seg
    ang = np.degrees(np.arctan2(y2-y1, x2-x1))
    return ang % 180.0  # [0,180)

def signed_distance(L, x, y):
    a,b,c = L
    return (a*x + b*y + c) / (np.hypot(a,b) + 1e-9)

def mean_normal(lines):
    """Average normal (a,b) of a set of normalized lines ax+by+c=0."""
    if not lines: 
        return np.array([0.0, 0.0], np.float32)
    v = np.mean([[L[0], L[1]] for L in lines], axis=0).astype(np.float32)
    n = np.linalg.norm(v); 
    return v / (n + 1e-9)

# --- NEW: corners via orientation clustering, checked with angle_between_lines ---
def find_rectangle_corners(segs, image_shape, expected_size, size_tol=0.30,
                           min_family_sep_deg=30,   # reject if the two families are too similar
                           use_edges_order=False):
    """
    segs: Nx4 segments from LSD (float32)
    image_shape: (H,W)
    expected_size: (expW, expH) in *pixels* at current resolution
    size_tol: +/- tolerance for size gate
    """
    if segs is None or len(segs) < 4:
        return None

    H, W = image_shape[:2]
    cx, cy = W*0.5, H*0.5

    # 1) Cluster segment orientations into two families (handles tilt/parallelogram)
    angs = np.array([[seg_angle(s)] for s in segs], np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, labels, centers = cv2.kmeans(angs, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    fam0 = [segs[i] for i,l in enumerate(labels.ravel()) if l == 0]
    fam1 = [segs[i] for i,l in enumerate(labels.ravel()) if l == 1]
    if not fam0 or not fam1: 
        return None

    # 2) Convert segments -> infinite normalized lines
    fam0_lines = [line_from_segment(s) for s in fam0]
    fam1_lines = [line_from_segment(s) for s in fam1]

    # 3) Use YOUR angle_between_lines() on mean normals to ensure near-orthogonal families
    n0 = mean_normal(fam0_lines)
    n1 = mean_normal(fam1_lines)
    # Build 3-coeff "lines" with these normals to reuse your angle_between_lines
    L0 = np.array([n0[0], n0[1], 0.0], np.float32)
    L1 = np.array([n1[0], n1[1], 0.0], np.float32)
    sep = angle_between_lines(L0, L1)  # 0 = parallel, 90 = perpendicular

    if sep < min_family_sep_deg or sep > 180 - min_family_sep_deg:
        # orientations too similar -> not a good rectangle
        return None

    # 4) Sort each family by signed distance from image center and take extremes
    fam0_sorted = sorted(fam0_lines, key=lambda L: signed_distance(L, cx, cy))
    fam1_sorted = sorted(fam1_lines, key=lambda L: signed_distance(L, cx, cy))
    if len(fam0_sorted) < 2 or len(fam1_sorted) < 2:
        return None

    # Decide which family we call "vertical-ish" vs "horizontal-ish"
    # If families are nearly orthogonal, either mapping works; pick based on which
    # yields a plausible size. We'll try both orders if needed.
    def corners_from_families(V_lines, H_lines):
        left,  right  = V_lines[0], V_lines[-1]
        top,   bottom = H_lines[0], H_lines[-1]
        tl = intersect(top, left);   tr = intersect(top, right)
        br = intersect(bottom, right); bl = intersect(bottom, left)
        if any(p is None for p in (tl,tr,br,bl)):
            return None
        box = np.array([tl, tr, br, bl], np.float32)
        return box

    # Try one order
    box = corners_from_families(fam0_sorted, fam1_sorted)
    if box is None or not size_is_valid(box, expected_size=expected_size, tol=size_tol):
        # Try swapped order
        box2 = corners_from_families(fam1_sorted, fam0_sorted)
        if box2 is not None and size_is_valid(box2, expected_size=expected_size, tol=size_tol):
            box = box2
        else:
            return None

    return box


def warp_from_corners(img, corners):
    rect = np.array(corners, np.float32)
    wA = np.linalg.norm(rect[0] - rect[1])
    wB = np.linalg.norm(rect[2] - rect[3])
    hA = np.linalg.norm(rect[0] - rect[3])
    hB = np.linalg.norm(rect[1] - rect[2])
    W = int(max(wA, wB))
    H = int(max(hA, hB))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    cx, cy = W//2, H//2
    cv2.line(warped, (cx,0), (cx,H), (0,255,0), 1)
    cv2.line(warped, (0,cy), (W,cy), (0,255,0), 1)
    return warped

def size_is_valid(box, expected_size=EXPECTED_SIZE, tol=SIZE_TOL):
    """
    Check if box dimensions are within expected size +/- tolerance.
    """
    wA = np.linalg.norm(box[0] - box[1])
    wB = np.linalg.norm(box[2] - box[3])
    hA = np.linalg.norm(box[0] - box[3])
    hB = np.linalg.norm(box[1] - box[2])
    W = max(wA, wB)
    H = max(hA, hB)

    expW, expH = expected_size
    if (expW*(1-tol) <= W <= expW*(1+tol)) and (expH*(1-tol) <= H <= expH*(1+tol)):
        return True
    return False

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
    corners = find_rectangle_corners(
        segs,
        image_shape=enhanced.shape,
        expected_size=EXPECTED_SIZE,
        size_tol=SIZE_TOL,
        min_family_sep_deg=20  # require families to differ by at least ~30°
    )
    print("Detected corners:", corners)
    vis = img.copy()
    if corners is not None:
        poly = corners.astype(int)
        cv2.polylines(vis, [poly], True, (0,255,0), 2)
        for (x,y) in poly:
            cv2.circle(vis, (int(x),int(y)), 5, (0,0,255), -1)
        warped = warp_from_corners(img, corners)
        cv2.imshow("Warped", warped)
    else:
        cv2.putText(vis, "No valid rectangle (size/orientation filter)", (20,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Corners", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
