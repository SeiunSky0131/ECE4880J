import cv2
import numpy as np
import os

# from .myEdgeDetector import myCanny
from myHoughLines import Handwrite_HoughLines
from myHoughTransform import Handwrite_HoughTransform

# parameters
sigma     = 2
threshold = 0.03
rhostep    = 2
thetastep  = np.pi / 90
num_lines    = 15
# end of parameters

img0 = cv2.imread('img02.jpg')
        
if (img0.ndim == 3):
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
else:
    img = img0

img_edge = cv2.Canny(img, 50, 150, apertureSize = 3)
img_edge = np.float32(img_edge) / 255
img_threshold = np.float32(img_edge > threshold)

[img_houghtrans, rhoList, thetaList] = Handwrite_HoughTransform(img_threshold, rhostep, thetastep)

cv2.imwrite('edge02.png', 255 * np.sqrt(img_edge / img_edge.max()))
cv2.imwrite('thres02.png', 255 * img_threshold)
cv2.imwrite('hough02.png', 255 * img_houghtrans / img_houghtrans.max())

[rhos, thetas] = Handwrite_HoughLines(img_houghtrans, num_lines)

# display your line segment results in red
for k in np.arange(num_lines):
    a = np.cos(thetaList[thetas[k]])
    b = np.sin(thetaList[thetas[k]])
    
    x0 = a*rhoList[rhos[k]]
    y0 = b*rhoList[rhos[k]]

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img0,(x1,y1),(x2,y2),(0,0,255),1)

lines = cv2.HoughLinesP(np.uint8(255 * img_threshold), rhostep, thetastep, \
                        50, minLineLength = 20, maxLineGap = 5)
# display line segment results from cv2.HoughLinesP in green
for line in lines:
    coords = line[0]
    cv2.line(img0, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 1)

cv2.imwrite('line02.png', img0)



