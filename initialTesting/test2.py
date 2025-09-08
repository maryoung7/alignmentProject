import cv2
import numpy as np

def detect_dark_mirror_and_lasers(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Threshold to detect dark regions (mirror is darker)
    _, dark_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, kernel)

    # Step 3: Find contours and select the largest as the mirror
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mirror_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            mirror_contour = cnt

    result = image.copy()

    if mirror_contour is not None:
        # Draw mirror contour
        cv2.drawContours(result, [mirror_contour], -1, (0, 255, 0), 2)

        # Step 4: Create a mask of the mirror region
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [mirror_contour], -1, 255, -1)

        # Step 5: Find laser dots (very bright spots inside mirror)
        mirror_gray = cv2.bitwise_and(gray, gray, mask=mask)
        _, laser_thresh = cv2.threshold(mirror_gray, 230, 255, cv2.THRESH_BINARY)

        # Optional: Erode to reduce noise
        laser_thresh = cv2.erode(laser_thresh, np.ones((3, 3), np.uint8), iterations=1)

        laser_contours, _ = cv2.findContours(laser_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in laser_contours:
            if cv2.contourArea(cnt) > 3:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        print("Mirror not found.")

    # Show results
    cv2.imshow("Gray", gray)
    cv2.imshow("Dark Threshold", dark_thresh)
    cv2.imshow("Detected Mirror and Lasers", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Use the uploaded image
detect_dark_mirror_and_lasers("Contour 8.png")
