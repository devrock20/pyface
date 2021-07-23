import cv2
import os
import random
os.system('fswebcam -r 1024x768 -S 10 --jpeg 50 --save /home/pi/Desktop/image.jpg')
imagePath = "/home/pi/Desktop/image.jpg"
cascPath = "/home/pi/Desktop/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.5,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    sub_face=image[y:y+h,x:x+w]
    name=random.randrange(1,1000)
    face_file_name="/home/pi/Desktop"+str(name)+".jpeg"
    cv2.imwrite(face_file_name,sub_face)
    

cv2.imshow("image", image)
cv2.waitKey(0)
