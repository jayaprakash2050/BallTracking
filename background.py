import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-vi", "--input",
    help="path to the input video file")
ap.add_argument("-vo", "--output",
    help="path to the output video file")
args = vars(ap.parse_args())


fourcc = cv2.VideoWriter_fourcc(*'MJPG')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
cap = cv2.VideoCapture(args['input'])
out = cv2.VideoWriter(args['output'], fourcc, 20.0, (640,400), True)
#createBackgroundSubtractorGMG
#createBackgroundSubtractorMOG
#createBackgroundSubtractorMOG2
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    if fgmask != None:
        cv2.imshow('frame',fgmask)
        out.write(frame)
    else: 
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()