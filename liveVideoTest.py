
#computerVision for a webcam/live video
import cv2
from PIL import Image
#read webcam (0 is default, change if multiple webcams)
webcam = cv2.VideoCapture(0)

#webcam doesn't end, so use while loop
while True:
    #ret = boolean if frame is read correctly
    #frame = the image from the webcam
    ret, frame = webcam.read()
    #convert to grayscale: cv2.COLOR_BGR2GRAY , hsv : cv2.COLOR_BGR2HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edge = cv2.Canny(gray, 100, 200)#mess around with these values

    #ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)#color below 80 is black, above is white

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('frame', thresh) #show the image in a window called 'frame'
    cv2.imshow('edge', edge) #show the image in a window called 'frame'
    #Image.fromarray(thresh).show()#show using PIL
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break #wait 40ms between frames, quit if q is pressed

webcam.release()#release the webcam when done
cv2.destroyAllWindows()#close all windows



#steps
#1. read webcam
#2. convert to grayscale
#3. adaptive threshold
#4. edge detection using canny
#5. display the edges over the original image