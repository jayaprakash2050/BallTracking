# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from random import randint
import numpy as np
import argparse
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (0, 0, 255)
#greenLower = (29, 86, 6)
greenUpper = (0, 0, 255)
#greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_tracked.avi', fourcc, 20.0, (640, 400), True)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
picks = []
flag = 0
xavg = 0
yavg = 0
xbvg = 0
ybvg = 0
check = True
count = 0
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	if check == True:
		orig = frame.copy()
		(rects, weights) = hog.detectMultiScale(orig, winStride=(4, 4),padding=(8, 8), scale=1.05)
		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		# draw the final bounding boxes
		if len(pick) > 0:
			(xA, yA, xB, yB) = pick[0]	
			if xA <= 450:	
				count+=1
				xavg = xavg + xA
				yavg = yavg + yA
				xbvg = xbvg + xB
				ybvg = ybvg + yB
				#print xA, 400-yA, xB, 400-yB
				if count == 10:
					check = False
					xavg = xavg / count
					yavg = yavg / count
					xbvg = xbvg / count
					ybvg = ybvg / count	
					#print "Xvalues: ", xavg, xbvg
	if xavg!=0 and count == 10:
		xcenter = (xavg + xbvg) / 2.0
		#print "center point: ", xcenter, 400-ybvg
		if randint(0,100)%2 == 0:
			cv2.rectangle(frame, (xavg+1, yavg+1), (xbvg+1, ybvg+1), (0, 255, 0), 2)
		else:
			cv2.rectangle(frame, (xavg-randint(0,5), yavg-randint(0,5)), (xbvg-randint(0,5), ybvg-randint(0,5)), (0, 255, 0), 2)
    ##
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=640)
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if True:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			#print x, ',' , (400-y)#
			if x > xavg - 50 and x < xbvg + 50:
				print x, ',' , (400-y)
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	frame_out = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
	out.write(frame_out)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
out.release()
cv2.destroyAllWindows()