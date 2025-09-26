import os
from PIL import Image
import cv2
import numpy as np


IMG_PATH = "img1.png"   # <- set this
sx = sy = 0.2
points = []
orig = cv2.imread(IMG_PATH)
resized = cv2.resize(orig, (0,0), fx=sx, fy=sy)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)   #convert to grayscale, necessary for thresholding

medBlur = cv2.medianBlur(gray,5) #blur image to reduce noise, could try Gaussian too
cv2.imshow('gray', medBlur)
thresh = cv2.adaptiveThreshold(medBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,27,1) #take adaptive threshold
medBlur = cv2.medianBlur(thresh,13) #blur image to reduce noise, could try Gaussian too
newimg = resized.copy() #copy of original image to draw contours on
imgcanny = resized.copy()
ret, thresh2 = cv2.threshold(thresh, 50,255,cv2.THRESH_BINARY_INV) #invert the previous threshold
cv2.imshow('medBlur', medBlur)
final = resized.copy()
cv2.imshow('thresh', thresh)
cv2.imshow('thresh2', thresh2)
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
image_path = os.path.join('img1.png')  # your path
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






cv2.imshow('Result', final)
cv2.waitKey(0)
cv2.destroyAllWindows()