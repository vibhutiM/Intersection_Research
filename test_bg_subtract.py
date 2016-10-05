
import numpy as np
import cv2


path = "D:\\NVIDIA Research\\Video Collection\\10th Floor Mudd\\GOPR0118.mp4"
cap = cv2.VideoCapture(path)

print cap.get(cv2.CAP_PROP_FPS)

fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.createBackgroundSubtractorKNN()
iter_err = 0
while(1):
    
    try:
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        open_kern = np.ones((5,5), np.uint8)
        #opening = cv2.dilate(fgmask,open_kern,iterations = 1)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, open_kern)

        #cv2.imshow('frame',fgmask)
        cv2.imshow('opening',opening)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    except Exception, e:
        print e
        iter_err = iter_err + 1
        if iter_err >=5:
            break
cap.release()
cv2.destroyAllWindows()

