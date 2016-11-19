#!/usr/local/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('bat.jpg')
blurred = cv2.GaussianBlur(img, (3, 3), 0)
gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
cv2.imshow("Frame", gray_image)
ret,thresh1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
v = np.median(blurred)
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))
edges = cv2.Canny(blurred, lower, upper)



_, contours, _ = cv2.findContours(edges, 1, 2)
cnt = contours[0]

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(edges,[box],0,(0,0,255),2)
cv2.imshow("Frame", thresh1)
key = cv2.waitKey(1) & 0xFF
raw_input()