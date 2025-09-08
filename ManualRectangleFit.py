import cv2
import numpy as np

#LOAD IMAGE  ----->  this will change to live feed from camera
img = cv2.imread("Contour 3.png") 


# Resize for easier processing
scale = 0.2   # 20% of original size
img = cv2.resize(img, (0,0), fx=scale, fy=scale)
H, W = img.shape[:2] # height and width of image, used later for trackbars

# Window
win_name = "Grating warping"
cv2.namedWindow(win_name)

# Trackbar helper
def nothing(x): pass # this does nothing, just a placeholder for trackbar callback

# Create sliders for length, width, pitch, yaw, roll, center x/y
cv2.createTrackbar("Length (px)", win_name, 200, 1200, nothing)
cv2.createTrackbar("Width (px)",  win_name, 150, 1200, nothing)
cv2.createTrackbar("Pitch",       win_name, 90, 180, nothing)   # Up/down (centered at 90)
cv2.createTrackbar("Yaw",         win_name, 90, 180, nothing)   # Left/right (centered at 90)
cv2.createTrackbar("Roll",        win_name, 90, 180, nothing)   # z rotation (centered at 90)
cv2.createTrackbar("Center X",    win_name, W//2,  max(1, W), nothing)
cv2.createTrackbar("Center Y",    win_name, H//2,  max(1, H), nothing)

# Button parameters (drawn on left image)
button_tl = (10, 10) # top-left corner
button_br = (110, 50) # bottom-right corner
button_color = (50, 50, 50)
button_text = "SET" # text on button

warped_result = None # holds warped image after pressing "Set"

# Compute rotation matrix
def get_rotation_matrix(pitch, yaw, roll):
    # Convert degrees to radians and center around 0 by subtracting 90 (so slider 90=0deg)
    p = np.deg2rad(pitch - 90.0)
    y = np.deg2rad(yaw   - 90.0)
    r = np.deg2rad(roll  - 90.0)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p),  np.cos(p)]
    ])
    Ry = np.array([
        [ np.cos(y), 0, np.sin(y)],
        [ 0,        1, 0       ],
        [-np.sin(y), 0, np.cos(y)]
    ])
    Rz = np.array([
        [ np.cos(r), -np.sin(r), 0],
        [ np.sin(r),  np.cos(r), 0],
        [ 0,          0,         1]
    ])
    
    return Ry @ Rx @ Rz

# Mouse callback to detect clicks on the "Set" button
def on_mouse(event, x, y, flags, param):
    global warped_result
    if event == cv2.EVENT_LBUTTONUP:
        # check if inside button rect
        if button_tl[0] <= x <= button_br[0] and button_tl[1] <= y <= button_br[1]:
            # compute homography and warp now
            # We call a helper that copies current trackbar settings and does the warp
            warped = compute_and_warp()
            warped_result = warped

cv2.setMouseCallback(win_name, on_mouse)

# Helper to compute projected polygon and perform the warp (returns warped image or None)
def compute_and_warp():
    # Read slider values
    length = max(1, cv2.getTrackbarPos("Length (px)", win_name))
    width  = max(1, cv2.getTrackbarPos("Width (px)", win_name))
    pitch  = cv2.getTrackbarPos("Pitch", win_name)
    yaw    = cv2.getTrackbarPos("Yaw", win_name)
    roll   = cv2.getTrackbarPos("Roll", win_name)
    cx     = cv2.getTrackbarPos("Center X", win_name)
    cy     = cv2.getTrackbarPos("Center Y", win_name)

    # 3D rectangle corners (local coordinates, z=0 plane)
    rect3d = np.array([
        [-length/2.0, -width/2.0, 0.0],
        [ length/2.0, -width/2.0, 0.0],
        [ length/2.0,  width/2.0, 0.0],
        [-length/2.0,  width/2.0, 0.0]
    ], dtype=np.float32)

    # Rotate
    R = get_rotation_matrix(pitch, yaw, roll)
    rect_rot = rect3d @ R.T  # (4,3)

    # Simple perspective projection parameters
    f = 1000.0  # focal length in pixels (tweakable)
    src_pts = []
    for (x3, y3, z3) in rect_rot:
        zshift = z3 + f
        if zshift == 0:
            zshift = 1e-6
        px = f * (x3 / zshift) + cx
        py = f * (y3 / zshift) + cy
        src_pts.append([px, py])
    src_pts = np.array(src_pts, dtype=np.float32)  # shape (4,2)

    # Check area / degeneracy
    area = cv2.contourArea(src_pts)
    if abs(area) < 1.0:
        print("Projected polygon is degenerate (area ~ 0). Cannot warp.")
        return None

    # Destination rectangle (fronto-parallel). We'll use the length/width sliders as the target size.
    # Make them integers >0
    dst_w = max(1, int(round(length)))
    dst_h = max(1, int(round(width)))
    dst_pts = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype=np.float32)

    # Compute homography: map src_pts -> dst_pts
    try:
        Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts) #changes perspective of image
    except Exception as e:
        print("Failed to compute homography:", e)
        return None

    # use perpective transform points to warp image
    warped = cv2.warpPerspective(img, Hmat, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

      # draw center lines on warped image to help alignment
    ch, cw = warped.shape[:2]
    cx_w, cy_w = cw // 2, ch // 2
    cv2.line(warped, (cx_w, 0), (cx_w, ch), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(warped, (0, cy_w), (cw, cy_w), (0, 255, 0), 1, cv2.LINE_AA)

    return warped

# Main loop
while True:
    # Read sliders (for drawing the left image every frame)
    length = max(1, cv2.getTrackbarPos("Length (px)", win_name))
    width  = max(1, cv2.getTrackbarPos("Width (px)", win_name))
    pitch  = cv2.getTrackbarPos("Pitch", win_name)
    yaw    = cv2.getTrackbarPos("Yaw", win_name)
    roll   = cv2.getTrackbarPos("Roll", win_name)
    cx     = cv2.getTrackbarPos("Center X", win_name)
    cy     = cv2.getTrackbarPos("Center Y", win_name)

    # Rectangle corners in 3D and rotate (same as compute_and_warp)
    rect3d = np.array([
        [-length/2.0, -width/2.0, 0.0],
        [ length/2.0, -width/2.0, 0.0],
        [ length/2.0,  width/2.0, 0.0],
        [-length/2.0,  width/2.0, 0.0]
    ], dtype=np.float32)
    R = get_rotation_matrix(pitch, yaw, roll)
    rect_rot = rect3d @ R.T

    # projection to 2D points
    f = 1000.0
    proj_pts = []
    for (x3, y3, z3) in rect_rot:
        zshift = z3 + f
        if zshift == 0:
            zshift = 1e-6
        px = f * (x3 / zshift) + cx
        py = f * (y3 / zshift) + cy
        proj_pts.append([int(round(px)), int(round(py))])
    proj_pts = np.array(proj_pts, dtype=np.int32)

    # Draw left image copy
    left = img.copy()
   
    # draw polygon outline in green
    cv2.polylines(left, [proj_pts], isClosed=True, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)

    # Draw "Set" button
    cv2.rectangle(left, button_tl, button_br, button_color, thickness=-1)
    cv2.putText(left, button_text, (button_tl[0]+10, button_tl[1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # If user has pressed Set, warped_result holds those results
    if warped_result is not None:
        warped = warped_result.copy()
        # Resize warped img to match left image height while preserving aspect ratio
        wh, ww = warped.shape[:2]
        target_h = left.shape[0]
        scale = target_h / wh
        new_w = int(round(ww * scale))
        warped_resized = cv2.resize(warped, (new_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        # Empty placeholder image until user presses Set
        warped_resized = np.ones_like(left) * 200
        cv2.putText(warped_resized, "Press SET to warp selected area", (20, left.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2, cv2.LINE_AA)

    # Combine left and right into one canvas
    combined = np.hstack([left, warped_resized])

    # Show
    cv2.imshow(win_name, combined)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC exit
        break           #exits loop and exits program
    elif key == ord('s'):
        # Allow 's' to also trigger the Set action
        warped_result = compute_and_warp()

cv2.destroyAllWindows()
