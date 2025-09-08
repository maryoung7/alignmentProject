import cv2
import numpy as np

IMG_PATH = "Contour 8.png"   # <- set this

points = []
orig = cv2.imread(IMG_PATH)
orig = cv2.resize(orig, (0,0), fx=0.3, fy=0.3)
if orig is None:
    raise SystemExit(f"Could not read image at {IMG_PATH}")

def order_points(pts):
    """Return points as [top-left, top-right, bottom-right, bottom-left]."""
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
    for p in pts:
        cv2.circle(vis, tuple(map(int, p)), 5, (0, 0, 255), -1)
    if len(pts) >= 2:
        cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 255, 0), 1)
    if len(pts) == 4:
        ordered = order_points(pts)
        cv2.polylines(vis, [ordered.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.putText(vis, "Click 4 corners (any order). 'r' reset, 'q' quit.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis

def warp_from_points(src_img, pts):
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

        # --- add crosshairs ---
    cx, cy = maxW // 2, maxH // 2
    cv2.line(warped, (cx, 0), (cx, maxH), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (maxW, cy), (0, 255, 0), 1)
    
    return warped

def on_mouse(event, x, y, flags, param):
    global points, show
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        show = draw_preview(orig, points[:4])
        cv2.imshow("Input", show)
        if len(points) == 4:
            warped = warp_from_points(orig, points)
            cv2.imshow("Warped", warped)

# main UI loop
show = draw_preview(orig, points)
cv2.imshow("Input", show)
cv2.setMouseCallback("Input", on_mouse)

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q') or key == 27:   # q or Esc
        break
    if key == ord('r'):
        points = []
        cv2.destroyWindow("Warped")
        show = draw_preview(orig, points)
        cv2.imshow("Input", show)

cv2.destroyAllWindows()

