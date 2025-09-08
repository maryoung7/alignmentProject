import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------- Laser Contour Detection Function with Parameters ----------
def process_image(image, canny1, canny2, block_size, c_val):
    orig = image.copy()

    # Resize for consistency
    img_resized = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))

    # Convert to grayscale & blur
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection for mirror/grating
    edges = cv2.Canny(blur, canny1, canny2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest rectangle (mirror)
    mirror_contour = None
    max_area = 0
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > max_area:
            max_area = area
            mirror_contour = approx

    output = img_resized.copy()
    if mirror_contour is None:
        cv2.putText(output, "No mirror detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output

    cv2.drawContours(output, [mirror_contour], -1, (0, 255, 0), 2)

    # Mask the mirror region
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [mirror_contour], -1, 255, -1)
    mirror_region = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    mirror_gray = cv2.cvtColor(mirror_region, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold for laser
    block_size = max(3, block_size | 1)  # Ensure odd and >=3
    thresh = cv2.adaptiveThreshold(mirror_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c_val)

    # Find laser contours
    laser_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in laser_contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    return output

# ---------- Trackbar Callback ----------
def update_display(*args):
    canny1 = cv2.getTrackbarPos("Canny Th1", "Adjust Parameters")
    canny2 = cv2.getTrackbarPos("Canny Th2", "Adjust Parameters")
    block_size = cv2.getTrackbarPos("Block Size", "Adjust Parameters")
    c_val = cv2.getTrackbarPos("C Value", "Adjust Parameters")

    processed = process_image(current_image, canny1, canny2, block_size, c_val)
    cv2.imshow("Adjust Parameters", processed)

# ---------- File Selection ----------
def select_image():
    global current_image
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return

    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    current_image = image

    # Create OpenCV window with trackbars
    cv2.namedWindow("Adjust Parameters", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Canny Th1", "Adjust Parameters", 50, 500, update_display)
    cv2.createTrackbar("Canny Th2", "Adjust Parameters", 150, 500, update_display)
    cv2.createTrackbar("Block Size", "Adjust Parameters", 11, 51, update_display)
    cv2.createTrackbar("C Value", "Adjust Parameters", 2, 20, update_display)

    update_display()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):  # Save result
            params = (
                cv2.getTrackbarPos("Canny Th1", "Adjust Parameters"),
                cv2.getTrackbarPos("Canny Th2", "Adjust Parameters"),
                cv2.getTrackbarPos("Block Size", "Adjust Parameters"),
                cv2.getTrackbarPos("C Value", "Adjust Parameters"),
            )
            result = process_image(current_image, *params)
            import time
            output_path = f"laser_result_{int(time.time())}.png"
            cv2.imwrite(output_path, result)
            messagebox.showinfo("Saved", f"Result saved to {output_path}")

# ---------- Tkinter GUI ----------
root = tk.Tk()
root.title("Laser Contour Detector")
root.geometry("300x150")

label = tk.Label(root, text="Select an image to detect the laser contour", wraplength=250)
label.pack(pady=20)

select_btn = tk.Button(root, text="Select Image", command=select_image)
select_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.quit)
exit_btn.pack()

root.mainloop()
