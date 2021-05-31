"""
OpenCV Python Tutorial
GUI Interactions in OpenCV

Read the story here:
https://analyticsindiamag.com/real-time-gui-interactions-with-opencv-in-python/

    
"""

import cv2
import os

os.chdir('C:/Users/Nandhini/Python2021')


# ---------------------------------------------------------------------
# KEYBOARD INTERACTIONS

img = cv2.imread('mimi.jpg')
img = cv2.resize(img,(320,240))

# display the original image
cv2.imshow('Original Image', img)
key = cv2.waitKey(0) & 0xFF

if key == ord('q'):
    cv2.destroyAllWindows()
    
elif key == ord('s'):
    # save the image as such
    cv2.imwrite('mimi_colour.jpg', img)
    cv2.destroyAllWindows()

elif key == ord('g'):
    # convert to grayscale and save it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('mimi_gray.jpg', gray)
    cv2.destroyAllWindows()
    
elif key == ord('t'):
    # write some text and save it
    text_image = cv2.putText(img, 'Miracles of OpenCV', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),3)
    cv2.imwrite('mimi_text.jpg', text_image)
    cv2.destroyAllWindows()
    
# ----------------------------

# display the saved colour image
clr_img = cv2.imread('mimi_colour.jpg')
cv2.imshow('Colour Image', clr_img)

# display the saved grayscale image
gray_img = cv2.imread('mimi_gray.jpg')
cv2.imshow('Grayscale Image', gray_img)

# display the image with Text on it
text_img = cv2.imread('mimi_text.jpg')
cv2.imshow('Image with Text', text_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

# read two images
img_1 = cv2.imread('karsten_1.jpg', 1)
img_2 = cv2.imread('karsten_2.jpg', 1)

# create a window
cv2.namedWindow("My Pet")

# define a null callback function for Trackbar
def null(x):
    pass

# create a trackbar 
# arguments: trackbar_name, window_name, default_value, max_value, callback_fn
cv2.createTrackbar("Switch_Image", "My Pet", 0, 1, null)

while True:
    # get Trackbar position
    pos = cv2.getTrackbarPos("Switch_Image", "My Pet")
    if pos == 0:
        cv2.imshow("My Pet", img_1)
    elif pos == 1:
        cv2.imshow("My Pet", img_2)
    
    key = cv2.waitKey(1) & 0xFF
    # press 'q' to quit the window
    if key == ord('q'):
        break
cv2.destroyAllWindows()

# ---------------------------------------------------------------------
# BGR Control using Trackbars
import numpy as np
import cv2

#create a blank image
img = np.zeros([200,350,3], np.uint8)
cv2.namedWindow('BGR')

# define a null callback function for Trackbar
def null(x):
    pass

# create three trackbars for B, G and R 
# arguments: trackbar_name, window_name, default_value, max_value, callback_fn
cv2.createTrackbar("B", "BGR", 0, 255, null)
cv2.createTrackbar("G", "BGR", 0, 255, null)
cv2.createTrackbar("R", "BGR", 0, 255, null)

while True:
    # read the Trackbar positions
    b = cv2.getTrackbarPos('B','BGR')
    g = cv2.getTrackbarPos('G','BGR')
    r = cv2.getTrackbarPos('R','BGR')
    img[:] = [b,g,r] 
    cv2.imshow('BGR', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# HSV Control using Trackbars

# read a colourful image
img = cv2.imread('claudio.jpg')
img = cv2.resize(img, (320,280))

# convert BGR image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define a null callback function for Trackbar
def null(x):
    pass

# create six trackbars for H, S and V - lower and higher masking limits 
cv2.namedWindow('HSV')
# arguments: trackbar_name, window_name, default_value, max_value, callback_fn
cv2.createTrackbar("HL", "HSV", 0, 180, null)
cv2.createTrackbar("HH", "HSV", 180, 180, null)
cv2.createTrackbar("SL", "HSV", 0, 255, null)
cv2.createTrackbar("SH", "HSV", 255, 255, null)
cv2.createTrackbar("VL", "HSV", 0, 255, null)
cv2.createTrackbar("VH", "HSV", 255, 255, null)

while True:
    # read the Trackbar positions
    hl = cv2.getTrackbarPos('HL','HSV')
    hh = cv2.getTrackbarPos('HH','HSV')
    sl = cv2.getTrackbarPos('SL','HSV')
    sh = cv2.getTrackbarPos('SH','HSV')
    vl = cv2.getTrackbarPos('VL','HSV')
    vh = cv2.getTrackbarPos('VH','HSV')
    
    # create a manually controlled mask
    # arguments: hsv_image, lower_trackbars, higher_trackbars
    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
    
    # derive masked image using bitwise_and method
    final = cv2.bitwise_and(img, img, mask=mask)
    
    # display image, mask and masked_image 
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Masked Image', final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

# read a colourful image
img_1 = cv2.imread('balls.jpg')
img_1 = cv2.resize(img_1, (320,210))
# display the image
cv2.imshow('Colour', img_1)

# read another image
img_2 = cv2.imread('julian.jpg')
img_2 = cv2.resize(img_2, (320,210))

def Mouse_Event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b = img[x,y,0]
        g = img[x,y,1]
        r = img[x,y,2]
    
        # change the colour of a portion of image
        coloured = img_2.copy()
        coloured[-50:-1, 0:320, :] = [b,g,r]
        coloured = cv2.putText(coloured, 'Colourful Holi!', (40,190), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
        cv2.imshow('Clicked Colour', coloured)

# set Mouse Callback method
cv2.setMouseCallback('Colour', Mouse_Event)

cv2.waitKey(0)
cv2.destroyAllWindows()


