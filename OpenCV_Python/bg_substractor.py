# import the necessary packages
import cv2
import time
import imutils
import argparse
import datetime
import numpy as np
 
# python bg_substractor.py --video ../videos/GOPR0195.MP4 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


# loop over the frames of the video
while True:

    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    if not grabbed:
        continue

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # apply background substractor mask to frame
    fgmask = fgbg.apply(frame)
    open_kern = np.ones((5,5), np.uint8)

    # no need to apply both operation at the same time
    opening = cv2.dilate(fgmask,open_kern,iterations = 1)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, open_kern)

    (_, cnts, _) = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)    

    print '-----'
    for i, cnt in enumerate(cnts):
        # if the contour is too small, ignore it
        if 200<cv2.contourArea(cnt)<5000:
            cv2.drawContours(frame,[cnt],0,(0,255,0),2)
            cv2.drawContours(opening,[cnt],0,255,-1)

            # add arrow from to countour 
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
            cv2.arrowedLine(frame, bottommost, topmost, (255,0,0), thickness=2)

    cv2.imshow("Ground", frame)    
    cv2.imshow("Segmentation", opening)


    key = cv2.waitKey(1) & 0xFF 
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()

