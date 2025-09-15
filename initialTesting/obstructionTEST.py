import cv2, numpy as np

#not great
#just testing line detection


# --- load & edges ---
img = cv2.imread("Contour 8.png")
resized = cv2.resize(img, (640, 480))   #.resize to make image smaller
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

medBlur = cv2.medianBlur(gray,5) #blur image to reduce noise, could try Gaussian too
cv2.imshow('medBlur', medBlur)
thresh = cv2.adaptiveThreshold(medBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,1.5) #take adaptive threshold
edges = cv2.Canny(thresh, 50, 150)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #a kernel is a small matrix used for image processing
#closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) #morph close to close gaps in lines
#cv2.imshow('closed', closed)
cv2.imshow('edges', edges)
# --- detect lines ---
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                        minLineLength=80, maxLineGap=10)
line_img = resized.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#print(lines)
#find intersections of lines to find corners of square
def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # parallel lines

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)
points = []
if lines is not None:   
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = compute_intersection(lines[i][0], lines[j][0])
            if pt is not None:
                points.append(pt)
                cv2.circle(line_img, pt, 5, (255, 0, 0), -1)  # draw intersection points

cv2.imshow('lines', line_img)

#draw square, fit to the corners found
drawing = gray.copy()
pts = []
print(points)
for p in points:
    pts.append(p)

#manually add missing points based on visual inspection
pts.append([450,400])
pts.append([500,200])
# convert to array when done
pts = np.array(pts, dtype=int)



for pt in pts:
    cv2.circle(drawing, pt, 5, (255, 0, 0), -1)  # draw intersection points



cv2.imshow('square', drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()