"""
OpenCV Python Tutorial

Read the story here:
https://analyticsindiamag.com/getting-started-with-opencv-in-python/ 


"""

import cv2
import os

os.chdir('C:/Users/Nandhini/Python2021')

# read an image
img = cv2.imread('daria.jpg', 0)

# display the image
cv2.imshow('Image Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write the image
cv2.imwrite('daria_copy.jpg', img)

# ---------------------------------------------------------------------
# Date-Time

from datetime import datetime
text = str(datetime.now())

# read a video from file
capture = cv2.VideoCapture('drive_6.mp4')
# display the read video file
while capture.isOpened():
    ret, frame = capture.read()
    # add date-time to the frames
    frame = cv2.putText(frame, text, (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    if not ret:
        break
    cv2.imshow('Video Window', frame)
    cv2.waitKey(30)
    
capture.release()
cv2.destroyAllWindows()


# ---------------------------------------------------------------------

# read a video and write it to another file

capture = cv2.VideoCapture('swan.mp4')
# get frame properties
print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(capture.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_1.avi', fourcc, 20.0, (640,360))

while capture.isOpened():
    ret, frame = capture.read()
    out.write(frame)
    if not ret:
        break
    cv2.imshow('Video Window', frame)
    cv2.waitKey(25)
    
capture.release()
out.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

# read a coloured image
img = cv2.imread('mason.jpg', 1)
print(img.shape)

# draw a green vertical line in it
# arguments: image, start, end, colour, thickness
img = cv2.line(img, (50,100), (50,300), (0,255,0), 10) 
# display the image
cv2.imshow('Image Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------

img = cv2.imread('mason.jpg', 1)

# draw a green vertical line in it
# arguments: image, start, end, colour, thickness
img = cv2.line(img, (50,100), (50,300), (0,255,0), 5) 

# draw a blue circle on it
# arguments: image, centre, radius, colour, thickness
img = cv2.circle(img, (150,250), 60, (255,0,0), 5) 

# draw a red rectangle on it
# arguments: image, diagonal_start, diagonal_end, colour, thickness
img = cv2.rectangle(img, (300,140), (400,270), (0,0,255), 5)

# display the image
cv2.imshow('Image Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

img = cv2.imread('senjuti.jpg')
img = cv2.resize(img,(300,450))

# display the  Original image without Text
cv2.imshow('Original Image', img)

text_image = cv2.putText(img, 'I love Colours', (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

# display the image with Text on it
cv2.imshow('Image with Text', text_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

# read a video from file
capture = cv2.VideoCapture('swan.mp4')
# display the read video file
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow('Video Window', frame)
    cv2.waitKey(30)
    
capture.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------------------
# coffee cup alteration

img = cv2.imread('brooke.jpg')
img = cv2.resize(img,(320,480))

coffee = img[150:235,200:300]
alter = img.copy()
alter[0:85,0:100] = coffee
alter[85:170,0:100] = coffee
alter[170:255,0:100] = coffee
alter[20:105,220:320] = coffee

# display the Original image
cv2.imshow('Original Image', img)

# display the altered image
cv2.imshow('Altered Image', alter)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

# read an image
img = cv2.imread('daria.jpg', 1)
img = cv2.resize(img, (320,225))
# split the colour image and merge back
B, G, R = cv2.split(img)
img_BGR = cv2.merge((B,G,R))
img_RGB = cv2.merge((R,G,B))
img_BRG = cv2.merge((B,R,G))

# display the images
cv2.imshow('Image in BGR', img_BGR)
cv2.imshow('Image in RGB', img_RGB)
cv2.imshow('Image in BRG', img_BRG)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------

# read two images and resize them
img_1 = cv2.imread('senjuti.jpg', 1)
img_1 = cv2.resize(img_1, (300,450))
img_2 = cv2.imread('adriano.jpg',1)
img_2 = cv2.resize(img_2, (300,450))

img_1 = cv2.imread('xuan.jpg', 1)
img_1 = cv2.resize(img_1, (320,225))
img_2 = cv2.imread('daria.jpg',1)
img_2 = cv2.resize(img_2, (320,225))

# display original images
cv2.imshow('Image 1', img_1)
cv2.imshow('Image 2', img_2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# simple addition
simple = cv2.add(img_1, img_2)
cv2.imshow('Simple Addition', simple)

cv2.waitKey(0)
cv2.destroyAllWindows()

# weighted addition
weight_70 = cv2.addWeighted(img_1, 0.7, img_2, 0.3, 0)
weight_30 = cv2.addWeighted(img_1, 0.3, img_2, 0.7, 0)
cv2.imshow('70-30 Addition', weight_70)
cv2.imshow('30-70 Addition', weight_30)


cv2.waitKey(0)
cv2.destroyAllWindows()
