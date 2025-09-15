import os
from PIL import Image
import cv2
import numpy as np

#Automatic contour detection and fitting a rectangle to it
#No perspective correction
#WORKING FOR CONTOUR 4 DO NOT CHANGE

#Could probably edit to make better if colors are clearer in the image

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


#try using mesh?
#--------------------------------------------------------------------------------------------------------
#edge detection and hough lines (not working that well)     --------> idea here is to find lines and then extend them to make a box
# edge1 = cv2.Canny(thresh2, 30, 200) #perform edge detection
# lines = cv2.HoughLinesP(edge1, 
#                         rho=1, 
#                         theta=np.pi/180, 
#                         threshold=100, 
#                         minLineLength=50, 
#                         maxLineGap=30)

# # Draw lines
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(imgcanny, (x1, y1), (x2, y2), (0, 255, 0), 2) 
# --------------------------------------------------------------------------------------------------------


# Find contours and filter for quadrilaterals
# cnts, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in cnts:
#     print(cv2.contourArea(cnt))
#     if cv2.contourArea(cnt) > 300:  # filter out small blobs
#         rect = cv2.minAreaRect(cnt)  # center, (w,h), angle
#         box = cv2.boxPoints(rect)
#         box = box.astype(int) 

#         cv2.drawContours(newimg, [box], 0, (0,0,255), 3)
#--------------------------------------------------------------------------------------------------------


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) #a kernel is a small matrix used for image processing
closed = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines


edges = cv2.Canny(closed, 50, 150) #canny performs edge detection

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #retr_external retrieves only the extreme outer contours

#cv2.imshow('edges', edges) #print edges for debugging
#cv2.imshow('closed', closed) #print closed for debugging

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



# Display results
#cv2.imshow('Original Resized', resized)
#cv2.imshow('thresh', thresh) 
#cv2.imshow('edge1', edge1) 
#cv2.imshow('imgcanny', imgcanny)

#cv2.imshow('newimg', newimg)


#PART 2 - Mask the grating area and find contours within it
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
#cv2.imshow('Adaptive Thresh on ROI', thresh6)
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



cv2.imshow('Result', final)
cv2.waitKey(0)
cv2.destroyAllWindows()