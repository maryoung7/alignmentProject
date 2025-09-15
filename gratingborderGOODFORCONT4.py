import os
from PIL import Image
import cv2
import numpy as np

#Automatic contour detection and fitting a rectangle to it
#No perspective correction
#WORKING FOR CONTOUR 4 DO NOT CHANGE

#Could probably edit to make better if colors are clearer in the image

#Part 1 : inital processing -----------------------------------------------------------------
image_path = os.path.join('Contour 4png.png')  #specify your image path here
img = cv2.imread(image_path) #cv2.imread reads the image

resized = cv2.resize(img, (640, 480))   #.resize to make image smaller
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)   #convert to grayscale, necessary for thresholding

medBlur = cv2.medianBlur(gray,5) #blur image to reduce noise, could try Gaussian too

thresh = cv2.adaptiveThreshold(medBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2) #take adaptive threshold

newimg = resized.copy() #copy of original image to draw contours on
imgcanny = resized.copy()
ret, thresh2 = cv2.threshold(thresh, 127,255,cv2.THRESH_BINARY_INV) #invert the previous threshold

final = resized.copy()


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) #a kernel is a small matrix used for image processing
closed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines


#-------Part 2 : Detection--------------------------------------------

edges = cv2.Canny(closed, 50, 150) #canny performs edge detection


# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #retr_external retrieves only the extreme outer contours


# If no contours found, exit
if len(contours) == 0:
    print("No contours found.")
    exit()

# Find the largest contour by area
largest_contour = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(largest_contour)  # ((center_x, center_y), (width, height), angle)
box = cv2.boxPoints(rect)  # Gets 4 corners of the rectangle
box = box.astype(np.intp)
print(box)
# draw the box on the original image
cv2.drawContours(final, [box], 0, (0, 255, 0), 2)

#box is the four corners, use this for perpective warp:




#-------Part 3 : mask grating and find bright spots within-----------------------------------------

mask = np.zeros_like(gray) #create a black mask the size of the image
pts = np.array(box, np.int32) #convert box points to numpy array
cv2.fillPoly(mask, [pts], 255) #fill the box area with white
#black mask is applied everywhere except the box area
roi = cv2.bitwise_and(gray, gray, mask=mask) #apply the mask to the grayscale image
#cv2.imshow('Masked ROI', roi) #roi = region of interest

blur = cv2.blur(roi,(10,10))
#cv2.imshow('Blurred ROI', blur)
thresh6 = cv2.adaptiveThreshold(
    blur, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    35,   # block size (odd number, neighborhood size)
    -15    # C: subtracts from mean (tune to control sensitivity)
)
contours2, _ = cv2.findContours(thresh6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours2:
    # Get the largest bright spot (in case of noise)
    largest_contour = max(contours2, key=cv2.contourArea)
        
        # Fit a circle around it
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
        
        # Draw the circle
    cv2.circle(final, center, radius, (0, 0, 255), 2)
    cv2.circle(final, center, 2, (0, 255, 0), -1)  # small green dot at center


import os
import cv2
import numpy as np

# ----------------------------
# Input + preprocessing
# ----------------------------
image_path = os.path.join('Contour 4png.png')  # your path
img = cv2.imread(image_path)
resized = cv2.resize(img, (640, 480))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
medBlur = cv2.medianBlur(gray, 5)

thresh = cv2.adaptiveThreshold(
    medBlur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    15, 2
)
_, thresh2 = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
closed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

# ----------------------------
# Contour → rectangle
# ----------------------------
edges = cv2.Canny(closed, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise SystemExit("No contours found.")

largest_contour = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(largest_contour)  # ((cx,cy),(w,h),angle)
box = cv2.boxPoints(rect).astype(np.float32)

detected = resized.copy()
cv2.drawContours(detected, [box.astype(int)], 0, (0,255,0), 2)
cv2.imshow("Detected", detected)

# ----------------------------
# Perspective warp
# ----------------------------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

rect_pts = order_points(box)
wA = np.linalg.norm(rect_pts[0] - rect_pts[1])
wB = np.linalg.norm(rect_pts[2] - rect_pts[3])
hA = np.linalg.norm(rect_pts[0] - rect_pts[3])
hB = np.linalg.norm(rect_pts[1] - rect_pts[2])
maxW = int(max(wA, wB))
maxH = int(max(hA, hB))

dst = np.array([
    [0, 0],
    [maxW-1, 0],
    [maxW-1, maxH-1],
    [0, maxH-1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect_pts, dst)
warped = cv2.warpPerspective(resized, M, (maxW, maxH))

# crosshairs at center
cx, cy = maxW//2, maxH//2
cv2.line(warped, (cx, 0), (cx, maxH), (0,255,0), 1)
cv2.line(warped, (0, cy), (maxW, cy), (0,255,0), 1)

# ----------------------------
# Laser spot detection inside warped
# ----------------------------
gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
t = np.percentile(gray_warp, 99.5)  # tune threshold
_, mask = cv2.threshold(gray_warp, int(t), 255, cv2.THRESH_BINARY)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if cnts:
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] > 0:
        sx = int(M["m10"]/M["m00"])
        sy = int(M["m01"]/M["m00"])
        cv2.circle(warped, (sx,sy), 5, (0,0,255), -1)
        cv2.line(warped, (cx,cy), (sx,sy), (0,255,255), 1)
        ex, ey = sx-cx, sy-cy
        msg = f"error: ({ex:+.1f}, {ey:+.1f}) px"
        cv2.putText(warped, msg, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255), 2, cv2.LINE_AA)

cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()




#-------Part 4 : show results-----------------------------------------




cv2.imshow('Result', final)
cv2.waitKey(0)
cv2.destroyAllWindows()