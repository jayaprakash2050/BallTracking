from __future__ import print_function
import numpy as np
import cv2
####
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils


fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
cap = cv2.VideoCapture('straight-drive.mov')
out = cv2.VideoWriter('straight-drive_op.avi', fourcc, 20.0, (640,400), True)
#createBackgroundSubtractorGMG
#createBackgroundSubtractorMOG
#createBackgroundSubtractorMOG2
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame1 = cap.read()
    if not ret:
        break;
    fgmask = fgbg.apply(frame1)
    frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #detect person
    # load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	#image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = frame1.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame1, winStride=(4, 4),padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
    '''
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		filename, len(rects), len(pick)))
    '''
	# show the output images
	#cv2.imshow("Before NMS", orig)
	#cv2.imshow("After NMS", image)
	#cv2.waitKey(0)

    ##
    
    cv2.imshow('frame',frame)
    out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
