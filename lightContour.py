import cv2

#Just automatic contour detection and fitting a circle to it


img = cv2.imread('Contour 4png.png')
resized = cv2.resize(img, (640, 480))
img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) #a kernel is a small matrix used for image processing
closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines
cv2.imshow('closed', closed)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
adaptive = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2) #take adaptive threshold
cv2.imshow('adaptive', adaptive)
if contours:
    # Get the largest bright spot (in case of noise)
    largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a circle around it
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
        
        # Draw the circle
    cv2.circle(resized, center, radius, (0, 0, 255), 2)
    cv2.circle(resized, center, 2, (0, 255, 0), -1)  # small green dot at center


cv2.imshow('img', resized)
#cv2.imshow('thresh', thresh)
#cv2.imshow('blur', blur)

cv2.waitKey(0)
cv2.destroyAllWindows()