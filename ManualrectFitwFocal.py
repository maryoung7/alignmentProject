import cv2
import numpy as np

#manual fitting of 3d rectangle to image
#with perspective correction
#has focal length slider
#AND perspective correction

# --- Load image ------------------------------------------------
img = cv2.imread("Contour 4png.png")  # Replace with your image path
if img is None:
    img = np.ones((2592, 1944, 3), dtype=np.uint8) * 255

H, W = img.shape[:2]

# --- Window setup ---
win_name = "3D Rectangle - Left: original | Right: warped"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # allow resizing
cv2.resizeWindow(win_name, 1200, 900)         # initial manageable size

def nothing(x): pass

# --- Trackbars ---
cv2.createTrackbar("Length (px)", win_name, 500, 2000, nothing)
cv2.createTrackbar("Width (px)",  win_name, 500, 2000, nothing)
cv2.createTrackbar("Pitch",       win_name, 90, 180, nothing)
cv2.createTrackbar("Yaw",         win_name, 90, 180, nothing)
cv2.createTrackbar("Roll",        win_name, 90, 180, nothing)
cv2.createTrackbar("Center X",    win_name, W//2, W, nothing)
cv2.createTrackbar("Center Y",    win_name, H//2, H, nothing)
cv2.createTrackbar("Focal Length",win_name, 1000, 3000, nothing)  # NEW slider

# --- Button ---
button_tl = (10, 10)
button_br = (110, 50)
button_color = (50, 50, 50)
button_text = "SET"

warped_result = None

# --- Rotation helper ---
def get_rotation_matrix(pitch, yaw, roll):
    p = np.deg2rad(pitch - 90)
    y = np.deg2rad(yaw - 90)
    r = np.deg2rad(roll - 90)
    Rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rz = np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    return Ry @ Rx @ Rz

# --- Mouse callback for SET button ---
def on_mouse(event, x, y, flags, param):
    global warped_result
    if event == cv2.EVENT_LBUTTONUP:
        if button_tl[0] <= x <= button_br[0] and button_tl[1] <= y <= button_br[1]:
            warped_result = compute_and_warp()

cv2.setMouseCallback(win_name, on_mouse)

# --- Compute homography and warp ---
def compute_and_warp():
    length = max(1, cv2.getTrackbarPos("Length (px)", win_name))
    width  = max(1, cv2.getTrackbarPos("Width (px)", win_name))
    pitch  = cv2.getTrackbarPos("Pitch", win_name)
    yaw    = cv2.getTrackbarPos("Yaw", win_name)
    roll   = cv2.getTrackbarPos("Roll", win_name)
    cx     = cv2.getTrackbarPos("Center X", win_name)
    cy     = cv2.getTrackbarPos("Center Y", win_name)
    f      = cv2.getTrackbarPos("Focal Length", win_name)

    # 3D rectangle corners
    rect3d = np.array([
        [-length/2.0, -width/2.0, 0.0],
        [ length/2.0, -width/2.0, 0.0],
        [ length/2.0,  width/2.0, 0.0],
        [-length/2.0,  width/2.0, 0.0]
    ], dtype=np.float32)

    # Rotate
    R = get_rotation_matrix(pitch, yaw, roll)
    rect_rot = rect3d @ R.T

    # Perspective projection
    src_pts = []
    for x3, y3, z3 in rect_rot:
        zshift = z3 + f
        if zshift == 0: zshift = 1e-6
        px = f * (x3 / zshift) + cx
        py = f * (y3 / zshift) + cy
        src_pts.append([px, py])
    src_pts = np.array(src_pts, dtype=np.float32)

    # Degeneracy check
    if abs(cv2.contourArea(src_pts)) < 1.0:
        return None

    dst_pts = np.array([
        [0, 0],
        [length-1, 0],
        [length-1, width-1],
        [0, width-1]
    ], dtype=np.float32)

    Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, Hmat, (int(length), int(width)),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Add center cross lines
    ch, cw = warped.shape[:2]
    cv2.line(warped, (cw//2, 0), (cw//2, ch), (0,255,0), 2, cv2.LINE_AA)
    cv2.line(warped, (0, ch//2), (cw, ch//2), (0,255,0), 2, cv2.LINE_AA)

    return warped

# --- Main loop ---
while True:
    length = max(1, cv2.getTrackbarPos("Length (px)", win_name))
    width  = max(1, cv2.getTrackbarPos("Width (px)", win_name))
    pitch  = cv2.getTrackbarPos("Pitch", win_name)
    yaw    = cv2.getTrackbarPos("Yaw", win_name)
    roll   = cv2.getTrackbarPos("Roll", win_name)
    cx     = cv2.getTrackbarPos("Center X", win_name)
    cy     = cv2.getTrackbarPos("Center Y", win_name)
    f      = cv2.getTrackbarPos("Focal Length", win_name)

    # Compute projected points
    rect3d = np.array([
        [-length/2.0, -width/2.0, 0.0],
        [ length/2.0, -width/2.0, 0.0],
        [ length/2.0,  width/2.0, 0.0],
        [-length/2.0,  width/2.0, 0.0]
    ], dtype=np.float32)
    R = get_rotation_matrix(pitch, yaw, roll)
    rect_rot = rect3d @ R.T

    proj_pts = []
    for x3, y3, z3 in rect_rot:
        zshift = z3 + f
        if zshift == 0: zshift = 1e-6
        px = f * (x3 / zshift) + cx
        py = f * (y3 / zshift) + cy
        proj_pts.append([int(round(px)), int(round(py))])
    proj_pts = np.array(proj_pts, dtype=np.int32)

    left = img.copy()
    cv2.polylines(left, [proj_pts], True, (0,255,0), 2, cv2.LINE_AA)

    # Draw button
    cv2.rectangle(left, button_tl, button_br, button_color, -1)
    cv2.putText(left, button_text, (button_tl[0]+10, button_tl[1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Warped panel
    if warped_result is not None:
        warped = warped_result.copy()
        wh, ww = warped.shape[:2]
        target_h = 600  # limit display height for combined view
        scale = target_h / wh
        new_w = int(round(ww * scale))
        warped_resized = cv2.resize(warped, (new_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        warped_resized = np.ones((600, 600,3), dtype=np.uint8)*200
        cv2.putText(warped_resized, "Press SET to warp", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2, cv2.LINE_AA)

    # Combine for display
    combined = np.hstack([cv2.resize(left, (600,600)), warped_resized])
    cv2.imshow(win_name, combined)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        warped_result = compute_and_warp()

cv2.destroyAllWindows()
