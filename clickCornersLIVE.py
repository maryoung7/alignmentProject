import cv2
import numpy as np

#Easier than the manual rectangle fit, just have to click the corners
#shows continuous live feed of the warped region

# --- config ---
CAM_INDEX = 0   # change if you have multiple cameras
points = []
roi_set = False

#reorder 4 corners to make the sides of a rectangle
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

def warp_from_points(frame, pts):
    rect = order_points(pts) #starting points

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
    warped = cv2.warpPerspective(frame, M, (maxW, maxH))

    # add crosshairs
    cx, cy = maxW // 2, maxH // 2
    cv2.line(warped, (cx, 0), (cx, maxH), (0, 255, 0), 1)
    cv2.line(warped, (0, cy), (maxW, cy), (0, 255, 0), 1)

    return warped

def mouse_callback(event, x, y, flags, param):
    global points, roi_set
    if event == cv2.EVENT_LBUTTONDOWN and not roi_set:
        points.append((x, y))
        if len(points) == 4:
            roi_set = True

cap = cv2.VideoCapture(CAM_INDEX)
cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # draw selected points
    for p in points:
        cv2.circle(display, p, 5, (0, 0, 255), -1)

    # draw polygon if ROI selected
    if len(points) >= 2:
        cv2.polylines(display, [np.array(points, np.int32)], False, (0, 255, 0), 1)
    if roi_set:
        ordered = order_points(points)
        cv2.polylines(display, [ordered.astype(np.int32)], True, (0, 255, 0), 2)

        warped = warp_from_points(frame, points)
        cv2.imshow("Warped", warped)

    cv2.putText(display, "Click 4 corners of object. Press 'r' reset, 'q' quit.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Live Feed", display)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:  # quit
        break
    if key == ord('r'):  # reset ROI
        points = []
        roi_set = False
        cv2.destroyWindow("Warped")

cap.release()
cv2.destroyAllWindows()
