import cv2
import numpy as np

# Load the image
img = cv2.imread("Contour 8.png")

# Resize for easier processing
img = cv2.resize(img, (640, 480))
#img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Window
cv2.namedWindow("3D Rectangle")

# Trackbar helper
def nothing(x): pass

# Sliders
cv2.createTrackbar("Length", "3D Rectangle", 200, 600, nothing)
cv2.createTrackbar("Width", "3D Rectangle", 100, 600, nothing)
cv2.createTrackbar("Pitch", "3D Rectangle", 90, 180, nothing)   # Up/down
cv2.createTrackbar("Yaw", "3D Rectangle", 90, 180, nothing)     # Left/right
cv2.createTrackbar("Roll", "3D Rectangle", 90, 180, nothing)    # Twist
cv2.createTrackbar("Center X", "3D Rectangle", 400, 800, nothing)
cv2.createTrackbar("Center Y", "3D Rectangle", 300, 600, nothing)

def get_rotation_matrix(pitch, yaw, roll):
    # Convert to radians, centered around 0
    pitch = np.deg2rad(pitch - 90)  # X-axis
    yaw   = np.deg2rad(yaw - 90)    # Y-axis
    roll  = np.deg2rad(roll - 90)   # Z-axis

    # Rotation around X (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # Rotation around Y (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Rotation around Z (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])

    # Apply in order: roll → pitch → yaw
    return Ry @ Rx @ Rz


while True:
    # Clone image
    display = img.copy()

    # Get trackbar values
    length = cv2.getTrackbarPos("Length", "3D Rectangle")
    width  = cv2.getTrackbarPos("Width", "3D Rectangle")
    pitch  = cv2.getTrackbarPos("Pitch", "3D Rectangle")
    yaw    = cv2.getTrackbarPos("Yaw", "3D Rectangle")
    roll   = cv2.getTrackbarPos("Roll", "3D Rectangle")
    cx     = cv2.getTrackbarPos("Center X", "3D Rectangle")
    cy     = cv2.getTrackbarPos("Center Y", "3D Rectangle")

    # Rectangle corners in 3D (centered at origin, flat on XY plane)
    rect = np.array([
        [-length/2, -width/2, 0],
        [ length/2, -width/2, 0],
        [ length/2,  width/2, 0],
        [-length/2,  width/2, 0]
    ])

    # Apply rotation
    R = get_rotation_matrix(pitch, yaw, roll)
    rect_rotated = rect @ R.T

    # Simple perspective projection
    f = 600  # focal length
    proj = []
    for x, y, z in rect_rotated:
        z += f  # push forward
        if z == 0: z = 1
        px = int(f * (x / z) + cx)
        py = int(f * (y / z) + cy)
        proj.append((px, py))

    # Draw green rectangle
    proj = np.array(proj, dtype=np.int32)
    cv2.polylines(display, [proj], isClosed=True, color=(0,255,0), thickness=2)

    # Show
    cv2.imshow("3D Rectangle", display)

    # Exit on ESC
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
