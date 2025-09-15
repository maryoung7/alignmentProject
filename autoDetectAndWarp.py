import os
import cv2
import numpy as np
#need to rework the detection settings
# ----------------------------
# Config
# ----------------------------
IMG_PATH = "Contour 4png.png"   # your image path
DISPLAY_SIZE = (640, 480)       # (w,h) for consistent processing/preview
SPOT_PERCENTILE = 99.6          # bright-spot threshold percentile (tune)
DEADBAND_PX = 2.0               # deadband for considering "centered"
MIN_QUAD_AREA = 3000            # reject tiny quads (in resized pixels)

# ----------------------------
# Helpers
# ----------------------------
def order_points(pts):
    """Return corners in order [tl, tr, br, bl] as float32."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_from_points(src_img, pts):
    """Perspective-warp the quadrilateral pts to an axis-aligned rectangle + draw crosshairs."""
    rect = order_points(pts)
    # compute output size from ordered edges
    wA = np.linalg.norm(rect[0] - rect[1])
    wB = np.linalg.norm(rect[2] - rect[3])
    hA = np.linalg.norm(rect[0] - rect[3])
    hB = np.linalg.norm(rect[1] - rect[2])
    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))
    maxW = max(1, maxW)
    maxH = max(1, maxH)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src_img, M, (maxW, maxH))

    # crosshairs
    cx, cy = maxW // 2, maxH // 2
    cv2.line(warped, (cx, 0), (cx, maxH), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (maxW, cy), (0, 255, 0), 1)
    return warped

def enhance_for_dark(bgr):
    """
    Gentle enhancement for dark frames:
    - convert to LAB, CLAHE on L channel
    - optional slight gamma
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    labc = cv2.merge([Lc, A, B])
    out = cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)
    # mild gamma to lift mids, keep highlights
    gamma = 0.9  # <1 brightens slightly; tune 0.8–1.0
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype(np.uint8)
    out = cv2.LUT(out, table)
    return out

def find_mirror_box(resized_bgr):
    """
    Robust mirror/grating rectangle detector for dark, noisy scenes.
    Returns:
        box (4x2 int) or None,
        debug dict with intermediate images
    """
    dbg = {}
    h, w = resized_bgr.shape[:2]

    # --- 1) Contrast lift (CLAHE on L) ---
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    enh = cv2.cvtColor(cv2.merge([Lc, A, B]), cv2.COLOR_LAB2BGR)
    dbg["enh"] = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)

    # --- 2) Edge emphasis: black-hat to highlight bright rims on dark bg ---
    gray = dbg["enh"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    dbg["blackhat"] = blackhat

    # --- 3) Gradients + auto Canny ---
    gx = cv2.Scharr(blackhat, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blackhat, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    v = np.median(mag)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(mag, lo, hi, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    dbg["edges"] = edges

    # --- 4) Contours → candidate quads ---
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def quad_score(poly):
        # rectangularity & right angles
        pts = poly.reshape(-1,2).astype(np.float32)
        if pts.shape[0] != 4: return -1
        # side lengths
        d = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
        rect_w = (d[0]+d[2])/2.0; rect_h = (d[1]+d[3])/2.0
        rectangularity = min(rect_w, rect_h) / (max(rect_w, rect_h) + 1e-6)
        # angles
        def angle(a,b,c):
            v1 = a-b; v2 = c-b
            cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        angs = [angle(pts[(i-1)%4], pts[i], pts[(i+1)%4]) for i in range(4)]
        right_angle_score = 1.0 - (np.mean(np.abs(np.array(angs)-90.0))/45.0)  # 1 best, 0 bad
        # centeredness
        cx = np.mean(pts[:,0]); cy = np.mean(pts[:,1])
        cen = 1.0 - (np.hypot(cx - w/2, cy - h/2) / (0.75 * np.hypot(w/2, h/2)))
        area = cv2.contourArea(poly)
        return area * max(0, rectangularity) * max(0, right_angle_score) * max(0, cen)

    best_poly = None; best_s = -1
    for c in cnts:
        if cv2.contourArea(c) < 2000:   # reject tiny
            continue
        # polygonal approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        # straightness: each edge close to a line
        ok = True
        pts = approx.reshape(-1,2)
        for i in range(4):
            p1, p2 = pts[i], pts[(i+1) % 4]
            seg = np.vstack([p1, p2]).astype(np.float32)
            # sample mid-point and check deviation to the line fit
            line = cv2.fitLine(seg, cv2.DIST_L2, 0, 0.01, 0.01)  # returns [vx,vy,x0,y0]
            vx, vy, x0, y0 = line.flatten()
            # distance of the opposite corner to the line as a crude straightness cue
            p_opp = pts[(i+2) % 4].astype(np.float32)
            dist = abs(vy*(p_opp[0]-x0) - vx*(p_opp[1]-y0))
            if dist > 15:  # pixels; tune if needed
                ok = False; break
        if not ok: 
            continue

        s = quad_score(approx)
        if s > best_s:
            best_s, best_poly = s, approx

    if best_poly is not None:
        box = order_points(best_poly.reshape(4,2)).astype(np.intp)
        dbg["quad"] = box.copy()
        return box, dbg

    # --- 5) Fallback: Hough lines → build rectangle from dominant directions ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=min(w,h)//4, maxLineGap=20)
    dbg["lines"] = np.zeros_like(edges)
    if lines is not None:
        # separate roughly vertical and horizontal lines
        horizontals, verticals = [], []
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = l
            ang = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(ang) < 20 or abs(abs(ang)-180) < 20:
                horizontals.append(l)
            elif abs(abs(ang)-90) < 20:
                verticals.append(l)
            cv2.line(dbg["lines"], (x1,y1), (x2,y2), 255, 1)

        def merge_lines(ls, axis=0):
            if not ls: return None
            # cluster by median coordinate
            coords = [((l[0]+l[2])//2, (l[1]+l[3])//2) for l in ls]
            vals = [c[axis] for c in coords]
            val = int(np.median(vals))
            return val

        xL = merge_lines(verticals, axis=0)
        xR = merge_lines(verticals, axis=0)
        yT = merge_lines(horizontals, axis=1)
        yB = merge_lines(horizontals, axis=1)

        # if we at least have two orthogonal boundaries, build a box near center
        if xL is not None and xR is not None and yT is not None and yB is not None:
            xL = max(0, min(xL, w-1)); xR = max(0, min(xR, w-1))
            yT = max(0, min(yT, h-1)); yB = max(0, min(yB, h-1))
            if xR - xL > 40 and yB - yT > 40:
                box = np.array([[xL,yT],[xR,yT],[xR,yB],[xL,yB]], dtype=np.intp)
                dbg["quad_hough"] = box.copy()
                return box, dbg

    # Nothing worked
    return None, dbg


def find_spot_centroid(warped_bgr, percentile=99.6):
    """
    Detect the brightest blob (robust to dark scenes).
    Strategy: percentile threshold on grayscale -> open -> largest contour -> centroid (moments).
    """
    if warped_bgr.size == 0:
        return None, None

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # Avoid saturation bias: clip highlights gently before percentile
    # (optional) gray = np.minimum(gray, 250).astype(np.uint8)

    t = np.percentile(gray, percentile)
    _, mask = cv2.threshold(gray, int(t), 255, cv2.THRESH_BINARY)

    # Clean small specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy), mask

# ----------------------------
# Main (single image demo)
# ----------------------------
def main():
    if not os.path.exists(IMG_PATH):
        raise SystemExit(f"Image not found: {IMG_PATH}")

    img = cv2.imread(IMG_PATH)
    if img is None:
        raise SystemExit("cv2.imread failed to load the image.")

    # Resize to a predictable processing size
    resized = cv2.resize(img, DISPLAY_SIZE)

    # Enhance for dark images (gentle)
    enhanced = enhance_for_dark(resized)

    # 1) Auto-detect mirror/grating box (your pipeline)
    box, dbg = find_mirror_box(enhanced)
    detected_view = enhanced.copy()
    if box is None:
        cv2.putText(detected_view, "Mirror not found", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Detected", detected_view)
        if "edges" in dbg:
            cv2.imshow("DBG edges", dbg["edges"])
           # cv2.imshow("DBG closed", dbg["closed"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Draw the detected rectangle in the raw view
    cv2.drawContours(detected_view, [box], 0, (0, 255, 0), 2)

    # 2) Perspective warp to top-down mirror view
    warped = warp_from_points(enhanced, box.astype(np.float32))

    # 3) Detect the laser spot in the warped view
    spot, mask = find_spot_centroid(warped, percentile=SPOT_PERCENTILE)

    # 4) Draw overlays + compute error to the center
    warped_vis = warped.copy()
    H, W = warped_vis.shape[:2]
    center = (W/2.0, H/2.0)
    if spot is not None:
        ex = float(spot[0] - center[0])
        ey = float(spot[1] - center[1])
        err = (ex, ey)
        # draw spot and error vector
        cv2.circle(warped_vis, (int(round(spot[0])), int(round(spot[1]))), 5, (0, 0, 255), -1)
        cv2.line(warped_vis, (int(round(center[0])), int(round(center[1]))),
                 (int(round(spot[0])), int(round(spot[1]))), (0, 255, 255), 1)
        msg = f"err px: ({ex:+.1f}, {ey:+.1f})"
        color = (0,255,0) if np.hypot(ex,ey) <= DEADBAND_PX else (0,165,255)
        cv2.putText(warped_vis, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        err = None
        cv2.putText(warped_vis, "Laser not found", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    # mark exact center again (on top)
    cv2.circle(warped_vis, (int(round(center[0])), int(round(center[1]))), 4, (0,255,0), -1)

    # 5) (Optional) Create a ROI-only threshold view to help tuning
    if mask is not None:
        cv2.imshow("Spot mask (warped)", mask)

    # Show results
    cv2.imshow("Detected", detected_view)
    cv2.imshow("Warped (top-down)", warped_vis)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6) Hook for motion control (later)
    # If 'err' is not None, map [ex, ey] -> small stage tilts using your calibrated Jacobian/inverse.
    # Example stub:
    # if err is not None:
    #     dtheta_x, dtheta_y = -kp * (G @ np.array([ex, ey]))
    #     send_relative_tilt(dtheta_x, dtheta_y)

if __name__ == "__main__":
    main()
