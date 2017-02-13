# import the necessary packages
import re
import cv2
import time
import imutils
import argparse
import datetime
import numpy as np
from os import walk, path


GRND_IMAGES = 1
CROP_IMAGES = 2
BOTH_IMAGES = 3

def check_for_files(base_path):

    f = []
    for (dirpath, dirnames, filenames) in walk(base_path):
        f.extend(filenames)
        break

    a = [int(re.split('[_.]',filename)[2]) for filename in filenames if '.jpg' in filename]

    if len(a) == 0: return 0
    else: return max(a) + 1
    
    


def main():

    # python bg_substractor.py --video ../videos/GOPR0195.MP4 
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    ap.add_argument("-t", "--type", default="cars", help="type of sample to capture")
    ap.add_argument("-i", "--index", type=int, default=0, help="name_index")
    ap.add_argument("-d", "--dir", default="../samples/", help="directory to store samples")
    ap.add_argument("-m", "--mode", type=int, default=3, help="directory to store samples")

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


    mode = args["mode"]
    base_path = args["dir"]
    base_name = args["type"]
    index = check_for_files(base_path)

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


        # read frame to get image features
        height, width, channels = frame.shape

        # apply background substractor mask to frame
        fgmask = fgbg.apply(frame)
        open_kern = np.ones((5,5), np.uint8)

        # no need to apply both operation at the same time
        opening = cv2.dilate(fgmask,open_kern,iterations = 1)
        opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, open_kern)

        (_, cnts, _) = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)    


        # keep the original image to obtain cropped portions
        original = frame.copy()
        # create black image to later save ground truth
        ground_truth = np.zeros((height,width,3), np.uint8)

        for i, cnt in enumerate(cnts):
            # if the contour is too small, ignore it
            if 200<cv2.contourArea(cnt)<5000:

                (x, y, w, h) = cv2.boundingRect(cnt)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(frame,[cnt],0,(0,255,0),2)

                if GRND_IMAGES == mode or BOTH_IMAGES == mode:                                    
                    cv2.drawContours(ground_truth,[cnt],0,(0,255,0),
                        thickness=cv2.FILLED)
                    
        cv2.imshow("Ground", frame)    
        cv2.imshow("Segmentation", opening)

        key = cv2.waitKey(1) & 0xFF 
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):        
            break

        # if the `s` key is pressed, save either cropped images or ground truth
        if key == ord("s"):
            screenshot_samples = 0
            subindex = 0

            # loop over each contour and get the cropped image
            if CROP_IMAGES == mode or BOTH_IMAGES == mode:                                    
                for i, cnt in enumerate(cnts):
                # if the contour is too small, ignore it
                    if 200<cv2.contourArea(cnt)<5000:                    
                    
                        # save the cropped image
                        subindex += 1
                        (x, y, w, h) = cv2.boundingRect(cnt)                
                        cv2.imwrite(base_path + base_name + '_cropped_' + str(index) + 
                                '_' + str(subindex) + ".jpg", original[y:y + h,x:x + w])                    

                        screenshot_samples += 1


            if GRND_IMAGES == mode or BOTH_IMAGES == mode:                                                    
                cv2.imwrite(base_path + base_name + '_bg_' + str(index) + ".jpg", ground_truth)

            cv2.imwrite(base_path + base_name + '_ground_'+ str(index) + ".jpg", frame)
            index += 1

            print("Screenshot taken")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()