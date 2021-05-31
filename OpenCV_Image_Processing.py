"""
OpenCV Python Tutorial

# IMAGE PROCESSING WITH OPENCV PYTHON

Read the story here:
https://analyticsindiamag.com/image-processing-with-opencv-in-python/

"""


import cv2
import os

os.chdir('C:/Users/Nandhini/Python2021')

# read an image in grayscale
img = cv2.imread('daria.jpg', 0)
img = cv2.resize(img, (320,225))

# apply various thresholds
val, th1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
val, th2 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
val, th3 = cv2.threshold(img, 110, 255, cv2.THRESH_TRUNC)
val, th4 = cv2.threshold(img, 110, 255, cv2.THRESH_TOZERO)
val, th5 = cv2.threshold(img, 110, 255, cv2.THRESH_TOZERO_INV)

# display the images
cv2.imshow('Original', img)
cv2.imshow('THRESH_BINARY', th1)
cv2.imshow('THRESH_BINARY_INV', th2)
cv2.imshow('THRESH_TRUNC', th3)
cv2.imshow('THRESH_TOZERO', th4)
cv2.imshow('THRESH_TOZERO_INV', th5)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------

img = cv2.imread('daria.jpg', 0)
img = cv2.resize(img, (320,225))

# apply various adaptive thresholds
th1 = cv2.adaptiveThreshold(img, 255, \
                                 cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 7, 4)
th2 = cv2.adaptiveThreshold(img, 255, \
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 7, 4)
# display the images
cv2.imshow('ADAPTIVE_THRESHOLD_MEAN', th1)
cv2.imshow('ADAPTIVE_THRESHOLD_GAUSSIAN', th2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------



img = cv2.imread('tree.png', 0)

# Apply median blur
img1 = cv2.medianBlur(img,3)

# display the images
cv2.imshow('Original', img)
cv2.imshow('Median', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------


img = cv2.imread('sharon.jpg', 1)
img = cv2.resize(img, (300,300))
# Apply blur
img1 = cv2.blur(img,(3,3))

# display the images
cv2.imshow('Original', img)
cv2.imshow('Blur', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------


img = cv2.imread('keiron.jpg', 1)
img = cv2.resize(img, (320,210))

# Apply Gaussian blur
img1 = cv2.GaussianBlur(img,(5,5),2)

# display the images
cv2.imshow('Original', img)
cv2.imshow('Gaussian', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------


img = cv2.imread('keiron.jpg', 1)
img = cv2.resize(img, (320,210))

# Apply Bilateral Filter
img1 = cv2.bilateralFilter(img,7,100,100)

# display the images
cv2.imshow('Original', img)
cv2.imshow('Bilateral', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------
import numpy as np

img = cv2.imread('chessboard.jpg', 0)
img = cv2.resize(img, (300,200))

# Laplacian image gradient
lap = np.uint8(np.absolute(cv2.Laplacian(img,cv2.CV_64F, ksize=1)))

# display the images
cv2.imshow('Original', img)
cv2.imshow('Lpalacian', lap)
cv2.waitKey(0)
cv2.destroyAllWindows()



img = cv2.imread('chessboard.jpg', 0)
img = cv2.resize(img, (300,200))

# Sobel image gradient
vertical = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F, 1,0, ksize=1)))
horizon = np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F, 0,1, ksize=1)))


# display the images
cv2.imshow('Vertical', vertical)
cv2.imshow('Horizontal', horizon)
cv2.waitKey(0)
cv2.destroyAllWindows()


Sobel = cv2.bitwise_or(vertical, horizon)
cv2.imshow('Sobel', Sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------------------------------------------

img = cv2.imread('chessboard.jpg', 0)
img = cv2.resize(img, (450,300))

def null(x):
    pass

# create trackbars to control threshold values
cv2.namedWindow('Canny')
cv2.resizeWindow('Canny', (450,300))
cv2.createTrackbar('MIN', 'Canny', 80,255, null)
cv2.createTrackbar('MAX', 'Canny', 120,255, null)

while True:
    # get Trackbar position
    a = cv2.getTrackbarPos('MIN', 'Canny')
    b = cv2.getTrackbarPos('MAX', 'Canny')
    
    # Canny Edge detection
    # arguments: image, min_val, max_val
    canny = cv2.Canny(img,a,b)

    # display the images
    cv2.imshow('Canny', canny)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()

# --------------------------------------------------------------


img = cv2.imread('valerie.jpg', 1)
img = cv2.resize(img, (320,480))
# show original image
cv2.imshow('Original', img)

# binary thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
val,th = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY)

# find contours
contours,_ = cv2.findContours(th, 
                              cv2.RETR_TREE,
                              cv2.CHAIN_APPROX_NONE)
# draw contours on original image

face = contours[455:465]
cv2.drawContours(img, face, -1, (0,0,255),1)

cv2.imshow('Contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(contours))

















