##from dragCornersLive.py

## accept input for grating corners
## calibration?
## perform warp
# git add .
# git commit -m "message"
# When done, git push
# --- Imports & constants --- #

import numpy as np
import cv2
sx = sy = 0.5          # scale of video
HANDLE_R = 10          # radius of corner circles
PICK_R2 = 15**2        # pick radius squared(how close mouse must be to grab)
PAD = 100              # fixed padding (workspace border)

# --- Camera utilities --- #

def open_camera(index):
    
    cap = cv2.VideoCapture(index) #try default backend, opens camera at specified index
    if cap is not None and cap.isOpened():#some camera is opened, yay it worked, return camera object
        return cap
    if cap is not None: #if it fails to open, release the object
        cap.release()
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) #try again for specific backend for Windows
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()
    return None #if both attampts fail, return None
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
    win = "Select Camera" 
    cv2.namedWindow(win) #creates window to allow user to select camera
    cur = int(max(0, min(max_index, initial))) #Current camera index, start at initial index(likely 0)
    cap = None # no camera yet

    def _release(): # ensure camera is properly released
        nonlocal cap
        if cap is not None:
            cap.release()
            cap = None

    while True: # runs until user selects camera or exits
        if cap is None:
            cap = open_camera(cur) #open current camera index

        ok, frame = (False, None) #ok : whether frame read was successful, frame : the actual frame
        if cap is not None:
            ok, frame = cap.read()

        if not ok or frame is None: #failed to read frame, show error message
            frame = np.full((360, 640, 3), 30, np.uint8)
            cv2.putText(frame, f"Camera {cur} not available",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        else: # successfully read frame
            if preview_scale != 1.0: # rezize frame for preview if needed
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
        
        if k in LEFT_KEYS:
            new = max(0, cur - 1) #decrease index, but not below 0
            if new != cur: #if index changed, update and release camera
                cur = new
                _release()
        if k in RIGHT_KEYS:
            new = min(max_index, cur + 1)
            if new != cur:
                cur = new
                _release()
        #loops to show the new frame


# --- Geometry & warp --- #
def draw_calibration_grid(img, nx=8, ny=8,
                          color=(0, 255, 0),
                          thickness=1,
                          center_thickness=2):
    """
    Draw a calibration grid on img.
    nx, ny: number of cells in x/y (lines = nx-1, ny-1).
    """
    h, w = img.shape[:2]

    # vertical lines
    for i in range(1, nx):
        x = int(i * w / nx)
        cv2.line(img, (x, 0), (x, h), color, thickness)

    # horizontal lines
    for j in range(1, ny):
        y = int(j * h / ny)
        cv2.line(img, (0, y), (w, y), color, thickness)
def order_points(pts): #order points as tl, tr, br, bl
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")
def warp_from_points(src_img, pts):
    if len(pts) != 4:
        return None
    rect = order_points(pts)
    wA = np.linalg.norm(rect[0] - rect[1])
    wB = np.linalg.norm(rect[2] - rect[3])
    hA = np.linalg.norm(rect[0] - rect[3])
    hB = np.linalg.norm(rect[1] - rect[2])
    maxW = max(1, int(max(wA, wB)))
    maxH = max(1, int(max(hA, hB)))
    if maxW < 5 or maxH < 5:
        return None  # or keep last good warped image
    dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src_img, M, (maxW, maxH))
    
    # crosshairs for reference
    draw_calibration_grid(warped, nx=8, ny=8)

    #cx, cy = maxW // 2, maxH // 2
    #cv2.line(warped, (cx, 0), (cx, maxH), (0, 255, 0), 1)
    #cv2.line(warped, (0, cy), (maxW, cy), (0, 255, 0), 1)
    return warped


# --- UI & drawing --- #
points = []            # 4 pts in canvas coords

drag_idx = None        # index of point being dragged
work = None            # current padded frame
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
# --- Contour Detection --- #
def detect_bright_spot(frame_bgr, debug=False):
    """
    Detect the brightest spot (e.g., laser) in a BGR frame.
    Returns (center, radius) or (None, None) if nothing is found.
    """
    # 1) grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 2) optional morphology to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 3) threshold to isolate bright areas
    # You can tweak 200 → 150/220 depending on your laser brightness
    _, thresh = cv2.threshold(closed, 200, 255, cv2.THRESH_BINARY)

    # 4) find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        cv2.imshow("laser_thresh", thresh)

    if not contours:
        return None, None  # no bright spot

    # 5) pick largest bright blob and fit a circle
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


# --- Mouse callback & global state --- #

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

#MAIN
#select camera
def main():
    global points, work, drag_idx

    cam_idx = select_camera_cv2(max_index=10, preview_scale=0.6, initial=0)
    if cam_idx is None: # no camera selected
        print("No camera selected. Exiting.")
        raise SystemExit(0)
    cap = open_camera(cam_idx)
    if cap is None or not cap.isOpened():
        print(f"Failed to open camera {cam_idx}. Exiting.")
        raise SystemExit(0)
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
            center, radius = detect_bright_spot(warped, debug=False)

            if center is not None:
                # draw the detected laser on the warped image
                cv2.circle(warped, center, radius, (0, 0, 255), 2)
                cv2.circle(warped, center, 2, (0, 255, 0), -1)

                # optional: display coordinates on the warped view
                txt = f"Laser: {center}"
                cv2.putText(
                    warped, txt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
            cv2.imshow("Warped", warped)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break #quit
        if key == ord('r'):
            points = []  # reset corners

    cap.release()
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()