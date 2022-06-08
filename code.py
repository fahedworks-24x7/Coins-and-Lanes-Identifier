# pip install opencv-python

import cv2 as cv
import numpy as np

img = cv.imread('CountBooks_BookShelf.jpg')

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = rescaleFrame(img)
cv.imshow('Rescaled', img)
cv.imwrite('Rescaled.jpg', img)

# Cropping
img = img[290:, :] 
cv.imshow('Cropped', img)
cv.imwrite('Cropped.jpg', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.imwrite('Gray.jpg', img)

# Blur
blur = cv.GaussianBlur(gray, (7,7), 0)
cv.imshow('Blur', blur)
cv.imwrite('Blur.jpg', img)

# Edge Cascade
canny = cv.Canny(blur, 50, 150, apertureSize=3)
cv.imshow('Canny Edges', canny)
cv.imwrite('Canny Edges.jpg', img)

# Eroding
eroded = cv.erode(canny, (5,5), iterations=3)
cv.imshow('Eroded', eroded)
cv.imwrite('Eroded.jpg', img)

lines = cv.HoughLinesP(image=eroded, rho=1,theta=np.pi/180, threshold=15,
            lines=np.array([]), minLineLength=120,maxLineGap=15)

a,b,c = lines.shape
for i in range(a):
    cv.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)

cv.imshow('Final_Image', img)
cv.imwrite('Final_Image.jpg', img)

print(len(lines))

cv.waitKey(0)