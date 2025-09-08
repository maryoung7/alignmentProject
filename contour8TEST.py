import os
from PIL import Image
import cv2
import numpy as np

##UI
#have input for the grating size

#based on grating size, estimate where the border should be using edges or sides of contour
#then draw a box there

#try hough transform to identify angles of interest
#look into tranforming image/ dewarping
#convex hull? - The convex hull of a contour is the smallest convex polygon that encloses the contour

image_path = os.path.join('Contour 8.png')  #specify your image path here
img = cv2.imread(image_path) #cv2.imread reads the image

resized = cv2.resize(img, (640, 480))   #.resize to make image smaller
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)   #convert to grayscale, necessary for thresholding

medBlur = cv2.medianBlur(gray,5) #blur image to reduce noise, could try Gaussian too
#cv2.imshow('medBlur', medBlur)

thresh = cv2.adaptiveThreshold(medBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,1) #take adaptive threshold
newimg = resized.copy() #copy of original image to draw contours on
imgcanny = resized.copy()
ret, thresh2 = cv2.threshold(thresh, 127,255,cv2.THRESH_BINARY_INV) #invert the previous threshold
blur2 = cv2.medianBlur(thresh2,7)
blur2 = cv2.GaussianBlur(blur2, (5, 5), 0)
cv2.imshow('blur2', blur2)

final = resized.copy()
cv2.imshow('thresh2', thresh2)

#--------------------------------------------------------------------------------------

# Detect edges
edges1 = cv2.Canny(img, 50, 150)

# Detect lines
lines = cv2.HoughLinesP(edges1, 1, np.pi/180, threshold=50,
                        minLineLength=50, maxLineGap=10)
#cv2.imshow('edges1', edges1)

# Collect intersection points
# points = []
# for i in range(len(lines)):
#     for j in range(i+1, len(lines)):
#         # Compute line intersection
#         pt = compute_intersection(lines[i][0], lines[j][0])
#         if pt is not None:
#             points.append(pt)

# If 3 corners found, reconstruct 4th
# if len(points) >= 3:
#     pts = np.array(points[:3], dtype=np.float32)
#     # Compute missing corner from square geometry
#     fourth = pts[0] + pts[2] - pts[1]  # parallelogram trick
#     pts = np.vstack([pts, fourth])
#--------------------------------------------------------------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)) #a kernel is a small matrix used for image processing
closed = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines


edges = cv2.Canny(closed, 50, 150) #canny performs edge detection
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) #a kernel is a small matrix used for image processing
closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines
# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #retr_external retrieves only the extreme outer contours
cv2.imshow('edges', edges) #print edges for debugging
cv2.imshow('closed', closed) #print closed for debugging

# If no contours found, exit
if len(contours) == 0:
    print("No contours found.")
    exit()

# Find the largest contour by area
largest_contour = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(largest_contour)  # ((center_x, center_y), (width, height), angle)
box = cv2.boxPoints(rect)  # Gets 4 corners of the rectangle
box = box.astype(np.intp) # convert to integer
print(box)
# draw the box on the original image
cv2.drawContours(final, [box], 0, (0, 255, 0), 2)



# Display results
#cv2.imshow('Original Resized', resized)
#cv2.imshow('thresh', thresh) 
#cv2.imshow('edge1', edge1) 
#cv2.imshow('imgcanny', imgcanny)

#cv2.imshow('newimg', newimg)
cv2.imshow('Result', final)
cv2.waitKey(0)
cv2.destroyAllWindows()