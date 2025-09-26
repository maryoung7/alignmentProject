import cv2, numpy as np



# --- load & edges ---
img = cv2.imread("img 3.png")
def photoProcessing(img):
    
    return
resized = cv2.resize(img, (640, 480))   #.resize to make image smaller
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
gx = cv2.Scharr(enhanced, cv2.CV_32F, 1, 0)
gy = cv2.Scharr(enhanced, cv2.CV_32F, 0, 1)
mag = cv2.magnitude(gx, gy)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

v = np.median(mag)
lo = int(max(0, 0.66*v))
hi = int(min(255, 1.33*v))
edges = cv2.Canny(mag, lo, hi, L2gradient=True)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

# Run directly on enhanced gray, or on edges if gray is too noisy
lines, _, _, _ = lsd.detect(enhanced)

overlay = resized.copy()
if lines is not None:
    for l in lines:
        x1,y1,x2,y2 = map(int, l[0])
        cv2.line(overlay, (x1,y1), (x2,y2), (0,255,255), 1, cv2.LINE_AA)

cv2.imshow("Detected lines", overlay)
#Now that we have line segments, try to find intersections/possible corners









#blur = cv2.GaussianBlur(enhanced, (0,0), 1.2)
#enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
cv2.imshow('enhanced', enhanced)
medBlur = cv2.medianBlur(enhanced,5) #blur image to reduce noise, could try Gaussian too
thresh = cv2.adaptiveThreshold(medBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,1.5) #take adaptive threshold
medBlur = cv2.medianBlur(thresh,9) #blur image to reduce noise, could try Gaussian too

cv2.imshow('thresh', thresh)
cv2.imshow('medBlur', medBlur)


edges = cv2.Canny(thresh, 50, 150)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #a kernel is a small matrix used for image processing
#closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines
#cv2.imshow('closed', closed)

# cv2.imshow('edges', edges)
# # --- detect lines ---
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
#                         minLineLength=80, maxLineGap=10)
# line_img = resized.copy()
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#print(lines)
#find intersections of lines to find corners of square
# def compute_intersection(line1, line2):
#     x1, y1, x2, y2 = line1
#     x3, y3, x4, y4 = line2

#     denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#     if denom == 0:
#         return None  # parallel lines

#     px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
#     py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
#     return int(px), int(py)
# points = []
# if lines is not None:   
#     for i in range(len(lines)):
#         for j in range(i+1, len(lines)):
#             pt = compute_intersection(lines[i][0], lines[j][0])
#             if pt is not None:
#                 points.append(pt)
#                 cv2.circle(line_img, pt, 5, (255, 0, 0), -1)  # draw intersection points

# cv2.imshow('lines', line_img)

# #draw square, fit to the corners found
# drawing = gray.copy()
# pts = []
# print(points)
# for p in points:
#     pts.append(p)

# #manually add missing points based on visual inspection
# pts.append([450,400])
# pts.append([500,200])
# # convert to array when done
# pts = np.array(pts, dtype=int)



# for pt in pts:
#     cv2.circle(drawing, pt, 5, (255, 0, 0), -1)  # draw intersection points



# cv2.imshow('square', drawing)


cv2.waitKey(0)
cv2.destroyAllWindows()