import numpy as np
import cv2


inputX = cv2.imread('test.jpg')
inputX = cv2.cvtColor(inputX,cv2.COLOR_BGR2GRAY)
xHeight = len(inputX)
xWidth = len(inputX[0])
gx = [[-3,0,3],
      [-10,0,10],
      [-3,0,3]]
gy = [[-3,-10,-3],
      [0,0,0],
      [3,10,3]]
yHeight = (len(inputX) - len(gx)) // 1 + 1
yWidth = (len(inputX[0]) - len(gx[0])) // 1 + 1
outputY1 = np.zeros((yHeight,yWidth))
outputY2 = np.zeros((yHeight,yWidth))
for i in range(yHeight):
      for j in range(yWidth):
            outputY1[i,j] = np.sum(inputX[i:i+3,j:j+3]*gx)
for i in range(yHeight):
      for j in range(yWidth):
            outputY2[i,j] = np.sum(inputX[i:i+3,j:j+3]*gy)
cv2.imshow('anh1',outputY1)
cv2.imshow('anh2',outputY2)
cv2.waitKey()
outputY1 = np.absolute(outputY1)
outputY2 = np.absolute(outputY2)
output = outputY1+outputY2
cv2.imshow('anh',output)
cv2.waitKey()




