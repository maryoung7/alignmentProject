import cv2
import numpy as np

def find_mirror_and_laser(image_path):
    # Load and resize image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    resized = cv2.resize(image, (640, 480))
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mirror_contour = None
    max_area = 0

    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 5000 and area > max_area:
                mirror_contour = approx
                max_area = area

    result = resized.copy()

    if mirror_contour is not None:
        # Draw mirror contour
        cv2.drawContours(result, [mirror_contour], -1, (0, 255, 0), 2)

        # Mask for inside mirror
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [mirror_contour], -1, 255, -1)

        # Apply mask and find bright spots (laser)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        _, thresh = cv2.threshold(masked, 220, 255, cv2.THRESH_BINARY)
        laser_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in laser_contours:
            if cv2.contourArea(cnt) > 10:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        print("No mirror detected.")

    # Show results
    cv2.imshow("Original", cv2.resize(resized, (400, 300)))
    cv2.imshow("Edges", cv2.resize(edges, (400, 300)))
    cv2.imshow("Result", cv2.resize(result, (400, 300)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Change this to your image path
image_path = "Contour 8.png"
find_mirror_and_laser(image_path)
