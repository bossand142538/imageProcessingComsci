#62050243
#Harit Chawalitbenja

import cv2
import numpy as np

widthImg = 800
heightImg = 600

#process right here Used to prepare an image before comparing it to a template.
#-------------------------------------------------------------------------------
def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>50:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
        cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def recorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#perspectivetransform
def getWarp(img,biggest):
    biggest = recorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return imgOutput
#-------------------------------------------------------------------------------

#1)Receive an image as the input
img = cv2.imread('1000.bmp')

#2)Locate the character and decode it
imgContour = img.copy()
cv2.resize(img, (widthImg, heightImg))
imgThres = preProcessing(img)
biggest = getContours(imgThres)
imgWarped = getWarp(img,biggest)
#After this, the image is ready to be compared to the template.

#create for Draw
imgWarpedDraw = imgWarped.copy()

haveNum = []
numLocate = []
numAndLocate = []


#3)Print the decoded number on the screen
for x in range(10):
    template = cv2.imread(str(x)+'.jpg')
    w, h, _ = template.shape

    #matchTemplate
    res = cv2.matchTemplate(imgWarped, template, cv2.TM_CCOEFF_NORMED)

    #adjust appropriate
    threshhold = 0.9
    loc = np.where(res >= threshhold)

    #This here to check and decodeใ
    #If the image is equal to the template Indicates that the number is present on the image.
    if not np.array_equal(loc, [[], []]):
        haveNum.append(x)
        yPosition, xPosition = loc
        numLocate.append(max(xPosition))
        numAndLocate.append([x,max(xPosition)])
        loc = [[yPosition[-1]],[xPosition[-1]]]

    #Draw line and text
    for pt in zip(*loc[::-1]):
        cv2.rectangle(imgWarpedDraw, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        cv2.putText(imgWarpedDraw, str(x), pt, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)

#sort number for print to screen
numAndLocate = sorted(numAndLocate,key=lambda l:l[1], reverse=False)

#print to screen
for i in numAndLocate:
    print(i[0],end = " ")

#and show as gui
cv2.imshow("original", img)
cv2.imshow("Result", imgWarpedDraw)
cv2.waitKey(0)
cv2.destroyAllWindow()

