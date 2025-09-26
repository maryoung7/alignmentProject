import cv2
import numpy as np

# --- config ---
CAM_INDEX = 0          # your webcam index
sx = sy = 0.5          # preview scale for speed; set to 1.0 for full res
HANDLE_R = 10
PICK_R2 = 15**2
PAD = 200              # fixed padding (workspace border)

# --- state (filled after first frame arrives) ---
points = []            # 4 pts in canvas coords
drag_idx = None
work = None            # current padded frame

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
    cv2.putText(vis, "Drag 4 red handles. 'r' reset, 'q' quit.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)
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
    # crosshairs
    cx, cy = maxW // 2, maxH // 2
    cv2.line(warped, (cx, 0), (cx, maxH), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (maxW, cy), (0, 255, 0), 1)
    return warped

def ensure_initialized(frame_padded):
    """Place 4 points around the live image area inside the padded canvas."""
    global points
    if len(points) == 4:
        return
    h, w = frame_padded.shape[:2]
    # live image occupies [PAD:PAD+iw, PAD:PAD+ih]
    ih, iw = h - 2*PAD, w - 2*PAD
    x0, y0 = PAD, PAD
    x1, y1 = PAD + iw, PAD + ih
    inset = int(0.08 * min(iw, ih))
    points = [
        (x0 + inset, y0 + inset),
        (x1 - inset, y0 + inset),
        (x1 - inset, y1 - inset),
        (x0 + inset, y1 - inset),
    ]
    points[:] = [tuple(map(float, p)) for p in points]

def nearest_handle(x, y, pts):
    if len(pts) != 4: return (None, float("inf"))
    q = np.array([x, y], dtype=np.float32)
    d2 = [float(np.sum((q - p)**2)) for p in pts]
    i = int(np.argmin(d2))
    return (i, d2[i]) if d2[i] <= PICK_R2 else (None, float("inf"))

def on_mouse(event, x, y, flags, param):
    global drag_idx, points, work
    if work is None:  # no frame yet
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        ensure_initialized(work)
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

# --- video setup ---
cap = cv2.VideoCapture(CAM_INDEX)
# (Optional) on Windows, try: cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # scale for speed
    if sx != 1.0 or sy != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=sx, fy=sy)

    # pad to get a fixed workspace
    work = cv2.copyMakeBorder(
        frame, PAD, PAD, PAD, PAD,
        borderType=cv2.BORDER_CONSTANT, value=(32, 32, 32)
    )

    ensure_initialized(work)

    # draw & show
    preview = draw_preview(work, points)
    cv2.imshow("Video", preview)

    # live warp
    if len(points) == 4:
        warped = warp_from_points(work, points)
        cv2.imshow("Warped", warped)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if key == ord('r'):
        points = []  # reinitialize next loop

cap.release()
cv2.destroyAllWindows()
