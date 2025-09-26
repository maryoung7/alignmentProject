import cv2
import numpy as np

# --- config ---
IMG_PATH = "img1.png"   # <- set this
sx = sy = 0.2
HANDLE_R = 10
PICK_R2 = 15**2
PAD = 80   # fixed padding around the image

# --- load image ---
orig = cv2.imread(IMG_PATH)
if orig is None:
    raise SystemExit(f"Could not read image at {IMG_PATH}")
orig = cv2.resize(orig, (0, 0), fx=sx, fy=sy)

# fixed padded canvas
work = cv2.copyMakeBorder(
    orig, PAD, PAD, PAD, PAD,
    borderType=cv2.BORDER_CONSTANT,
    value=(32, 32, 32)  # dark grey background
)

# --- state ---
points = []      # 4 pts in canvas coords
drag_idx = None

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def draw_preview(img, pts):
    vis = img.copy()
    if len(pts) == 4:
        ordered = order_points(pts)
        cv2.polylines(vis, [ordered.astype(np.int32)], True, (0, 255, 0), 2)
        for i, p in enumerate(ordered):
            p = tuple(map(int, p))
            cv2.circle(vis, p, HANDLE_R, (0, 0, 255), -1)
            cv2.circle(vis, p, HANDLE_R, (255, 255, 255), 1)
            cv2.putText(vis, str(i), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    else:
        for p in pts:
            cv2.circle(vis, tuple(map(int, p)), 5, (0, 0, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 255, 0), 1)

    cv2.putText(vis, "Drag handles inside grey border. 'r' reset, 'q' quit.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return vis

def warp_from_points(src_img, pts):
    rect = order_points(pts)
    wA = np.linalg.norm(rect[0] - rect[1])
    wB = np.linalg.norm(rect[2] - rect[3])
    hA = np.linalg.norm(rect[0] - rect[3])
    hB = np.linalg.norm(rect[1] - rect[2])
    maxW = max(1, int(max(wA, wB)))
    maxH = max(1, int(max(hA, hB)))

    dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src_img, M, (maxW, maxH))

    cx, cy = maxW // 2, maxH // 2
    cv2.line(warped, (cx, 0), (cx, maxH), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (maxW, cy), (0, 255, 0), 1)
    return warped

def ensure_initialized():
    """Place 4 points around the image area inside the canvas."""
    global points
    if len(points) == 4:
        return
    ih, iw = orig.shape[:2]
    x0, y0 = PAD, PAD
    x1, y1 = PAD + iw, PAD + ih
    inset = int(0.08 * min(iw, ih))
    points = [
        (x0 + inset, y0 + inset),
        (x1 - inset, y0 + inset),
        (x1 - inset, y1 - inset),
        (x0 + inset, y1 - inset),
    ]
    points = [tuple(map(float, p)) for p in points]

def nearest_handle(x, y, pts):
    if len(pts) != 4: return (None, float("inf"))
    q = np.array([x, y], dtype=np.float32)
    d2 = [float(np.sum((q - p)**2)) for p in pts]
    i = int(np.argmin(d2))
    return (i, d2[i]) if d2[i] <= PICK_R2 else (None, float("inf"))

def on_mouse(event, x, y, flags, param):
    global drag_idx, points
    if event == cv2.EVENT_LBUTTONDOWN:
        ensure_initialized()
        i, d2 = nearest_handle(x, y, points)
        drag_idx = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            h, w = work.shape[:2]
            nx = max(0, min(w - 1, x))
            ny = max(0, min(h - 1, y))
            points[drag_idx] = (float(nx), float(ny))
            cv2.imshow("Input", draw_preview(work, points))
            cv2.imshow("Warped", warp_from_points(work, points))
    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None

# --- main loop ---
ensure_initialized()
cv2.imshow("Input", draw_preview(work, points))
cv2.setMouseCallback("Input", on_mouse)
cv2.imshow("Warped", warp_from_points(work, points))

while True:
    key = cv2.waitKey(20) & 0xFF
    if key in (ord('q'), 27):
        break
    if key == ord('r'):
        points = []
        ensure_initialized()
        cv2.imshow("Input", draw_preview(work, points))
        cv2.imshow("Warped", warp_from_points(work, points))

cv2.destroyAllWindows()
