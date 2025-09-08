import cv2
import numpy as np
import os
from datetime import datetime

def detect_mirror_and_lasers(image_path, output_dir="outputs"):
    # Load and resize image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    dark_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mirror_contour = max(contours, key=cv2.contourArea) if contours else None

    result = image.copy()

    if mirror_contour is not None:
        cv2.drawContours(result, [mirror_contour], -1, (0, 255, 0), 2)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [mirror_contour], -1, 255, -1)
        mirror_region = cv2.bitwise_and(gray, gray, mask=mask)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 100
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mirror_region)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(result, (x, y), radius, (0, 0, 255), 2)
    else:
        print("Mirror not found.")

    # Generate timestamped filename
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)

    # Save image
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")

    # Optional display
    cv2.imshow("Detected Mirror and Lasers", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
detect_mirror_and_lasers("Contour 8.png")
