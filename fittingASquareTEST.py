import cv2
import numpy as np

#Manual fitting of 3d rectangle to image
#no perspective correction
#no roll

# Load the image
img = cv2.imread("Contour 8.png")

# Resize for easier processing
resized = cv2.resize(img, (640, 480))
img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)



# Window
cv2.namedWindow("3D Rectangle")

# Trackbars
def nothing(x): pass

cv2.createTrackbar("Length", "3D Rectangle", 200, 600, nothing)
cv2.createTrackbar("Width", "3D Rectangle", 100, 600, nothing)
cv2.createTrackbar("Pitch", "3D Rectangle", 0, 180, nothing)   # Up/down
cv2.createTrackbar("Yaw", "3D Rectangle", 0, 180, nothing)     # Left/right
cv2.createTrackbar("Center X", "3D Rectangle", 400, 800, nothing)
cv2.createTrackbar("Center Y", "3D Rectangle", 300, 600, nothing)

def get_rotation_matrix(pitch, yaw):
    # Convert degrees to radians
    pitch = np.deg2rad(pitch - 90)  # center around 0
    yaw = np.deg2rad(yaw - 90)

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

    return Ry @ Rx  # Apply X then Y rotation


while True:
    # Clone image
    display = img.copy()

    # Get trackbar values
    length = cv2.getTrackbarPos("Length", "3D Rectangle")
    width = cv2.getTrackbarPos("Width", "3D Rectangle")
    pitch = cv2.getTrackbarPos("Pitch", "3D Rectangle")
    yaw = cv2.getTrackbarPos("Yaw", "3D Rectangle")
    cx = cv2.getTrackbarPos("Center X", "3D Rectangle")
    cy = cv2.getTrackbarPos("Center Y", "3D Rectangle")

    # Rectangle corners in 3D (centered at origin, z=0 plane)
    rect = np.array([
        [-length/2, -width/2, 0],
        [ length/2, -width/2, 0],
        [ length/2,  width/2, 0],
        [-length/2,  width/2, 0]
    ])

    # Apply rotation
    R = get_rotation_matrix(pitch, yaw)
    rect_rotated = rect @ R.T

    # Simple perspective projection
    f = 600  # focal length
    proj = []
    for x, y, z in rect_rotated:
        z += f  # shift forward
        if z == 0: z = 1
        px = int(f * (x / z) + cx)
        py = int(f * (y / z) + cy)
        proj.append((px, py))

    # Draw rectangle
    proj = np.array(proj, dtype=np.int32)
    cv2.polylines(display, [proj], isClosed=True, color=(0,0,255), thickness=2)

    # Show
    cv2.imshow("3D Rectangle", display)

    # Exit on ESC
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()


