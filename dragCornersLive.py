import cv2
import numpy as np

#Good for testing for live video
#how to update git repo:
# git add .
# git commit -m "message"
# When done, git push
#add bright spot detection after for testing
# ---------- Simple OpenCV Camera Picker (no trackbar) ----------
def _open_cap(idx):
    """Try default backend; if it fails, try CAP_DSHOW (often needed on Windows)."""
    cap = cv2.VideoCapture(idx) #try default backend
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) #Windows fallback
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()
    return None #if no camera found
def fit_text(img, text, base_scale=0.6, max_width_ratio=0.95):
    h, w = img.shape[:2]
    scale = base_scale
    while True:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw <= int(w * max_width_ratio) or scale <= 0.3:
            return scale
        scale -= 0.05
def select_camera_cv2(max_index=10, preview_scale=0.6, initial=0):
    """
    Open a window that previews the current index.
    Controls:
      <- / -> : change camera index
      Enter / Space : select current index
      q / Esc : cancel (returns None)
    """
    win = "Select Camera" #create window
    cv2.namedWindow(win)
    cur = int(max(0, min(max_index, initial))) #go to initial index
    cap = None # no camera yet

    def _release():
        nonlocal cap
        if cap is not None:
            cap.release()
            cap = None

    while True:
        if cap is None:
            cap = _open_cap(cur) #open current camera index

        ok, frame = (False, None)
        if cap is not None:
            ok, frame = cap.read()

        if not ok or frame is None:
            # Show error canvas
            frame = np.full((360, 640, 3), 30, np.uint8)
            cv2.putText(frame, f"Camera {cur} not available",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            if preview_scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=preview_scale, fy=preview_scale)

        # UI overlay
        h, w = frame.shape[:2]
        help1 = f"[{cur}]  <-/-> change   Enter/Space select   q/Esc exit"
        cv2.putText(frame, "Camera Picker", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        scale = fit_text(frame, help1, base_scale=0.5)
        cv2.putText(frame, help1, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (230,230,230), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        LEFT_KEYS  = {81, 2424832}     # 81 (Linux/Mac), 2424832 (Windows)
        RIGHT_KEYS = {83, 2555904}     # 83 (Linux/Mac), 2555904 (Windows)
        k = cv2.waitKey(1) & 0xFFFFFFFF

        if k in (13, 32):  # Enter or Space
            _release()
            cv2.destroyWindow(win)
            return cur
        if k in (ord('q'), 27):  # q or Esc
            _release()
            cv2.destroyWindow(win)
            return None
        
        if k == LEFT_KEYS:
            new = max(0, cur - 1)
            if new != cur:
                cur = new
                _release()
        if k == RIGHT_KEYS:
            new = min(max_index, cur + 1)
            if new != cur:
                cur = new
                _release()

# --- camera selection UI ---
cam_idx = select_camera_cv2(max_index=10, preview_scale=0.6, initial=0)
if cam_idx is None: # no camera selected
    print("No camera selected. Exiting.")
    raise SystemExit(0)

cap = cv2.VideoCapture(cam_idx) #reopen camera that is chosen
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)  # Windows fallback
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cam_idx}")


# --- state / tuning constants ---

sx = sy = 0.5          # scale of video
HANDLE_R = 10          # radius of corner circles
PICK_R2 = 15**2        # pick radius squared(how close mouse must be to grab)
PAD = 100              # fixed padding (workspace border)

points = []            # 4 pts in canvas coords
drag_idx = None        # index of point being dragged
work = None            # current padded frame

def order_points(pts): #order points as tl, tr, br, bl
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def draw_preview(img, pts):#drawing of preview with handles
    vis = img.copy()
    if len(pts) == 4:
        ordered = order_points(pts)
        cv2.polylines(vis, [ordered.astype(np.int32)], True, (0, 255, 0), 2)
        for i, p in enumerate(ordered):
            p = tuple(map(int, p))
            cv2.circle(vis, p, HANDLE_R, (0, 0, 255), -1)
            cv2.circle(vis, p, HANDLE_R, (255, 255, 255), 1)
            cv2.putText(vis, str(i), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(vis, "Drag 4 red handles. 'r' reset, 'q' quit.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1, cv2.LINE_AA)
    return vis


#perspective warp from 4 points using OpenCV
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
    
    # crosshairs for reference
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

def nearest_handle(x, y, pts): #finds the nearest handle to current x,y
    if len(pts) != 4: return (None, float("inf"))
    q = np.array([x, y], dtype=np.float32)
    d2 = [float(np.sum((q - p)**2)) for p in pts]
    i = int(np.argmin(d2))
    return (i, d2[i]) if d2[i] <= PICK_R2 else (None, float("inf"))

def on_mouse(event, x, y, flags, param): #mouse callback
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

cv2.namedWindow("Video") #main window
cv2.setMouseCallback("Video", on_mouse) #mouse callback for this window

while True:
    ok, frame = cap.read()
    if not ok or frame is None: #keeps UI alive if camera fails
        frame = np.full((360, 640, 3), 30, np.uint8)
        cv2.putText(frame, "Camera stream lost (q/Esc to quit)",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
    #if not ok:
    #    break

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
        break #quit
    if key == ord('r'):
        points = []  # reset corners

cap.release()
cv2.destroyAllWindows()
