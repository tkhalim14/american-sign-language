from sre_constants import SUCCESS
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import math
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20 # for the hand frame for classifier
imgSize = 300

folder = "Data/J"
counter=0

#Teachable Machine.com for training models

while True:
    success,img = cap.read()
    hands, img = detector.findHands(img)
    if hands: # check if there is anything in the hands
        hand= hands[0]
        x,y,w,h = hand['bbox'] # x = min x coordinate, y = min y coordinate, w = width , h = height
        #x=max(y-offset:y+h+offset,x-offset:x+w+offset)
        cropimg = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        #print(cropimg.shape)
        aspectRatio = h/w

        if aspectRatio>1 :
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(cropimg, (wCal,imgSize))
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
        else :
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(cropimg, (imgSize,hCal))
            hGap = math.ceil((imgSize-hCal)/2)
            #imgWhite[0:imgResize.shape[0],0:imgResize.shape[1]] = imgResize
            imgWhite[hGap:hGap+hCal,:] = imgResize
        #cv2.imshow("ImageOverlay",imgWhite)
        cv2.imshow("CroppedImage",cropimg)
        cv2.imshow("Image",imgWhite)
    #time.sleep(0.019)
    if cv2.waitKey(1) & 0xFF == ord('q') :    #binary conversion
        break
    elif cv2.waitKey(1) & 0xFF == ord('s') :    #binary conversion
        counter+=1
        cv2.imwrite(f'{folder}/img_{counter}.jpg',imgWhite)
        print(counter)
cap.release()
cv2.destroyAllWindows()
