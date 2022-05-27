import cv2 
import numpy as np 
import os 
import sys

#def of lower/upper bounds of overlay colours for HSV Red"
lower_red = np.array([170,100,100], dtype = "uint16")
upper_red = np.array([180,255,255], dtype = "uint16")

#def of lower/upper bounds of overlay colours for HSV Green"
lower_green = np.array([60,90,90], dtype = "uint16")
upper_green = np.array([80,255,255], dtype = "uint16")

def main(video, save_path):
    START = True	
    cap = cv2.VideoCapture(video)
    rect = None
    count = 59
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    while cap.isOpened():

        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        _, img = cap.read()
        img = cv2.resize(img, (1071, 604), interpolation = cv2.INTER_LANCZOS4) #Increase Size 
        
        if START:
            print("Select second box")
            rect = cv2.selectROI("Image", img, showCrosshair=False, fromCenter=False)
            START = False

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

        if key == ord('f'):
            frame_count += 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if key == ord('b'):
            frame_count -= 10
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if key == ord('x'):
            bw_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(rect)
            print(bw_frame)
            cv2.imwrite(os.path.join(save_path,f'minutes_{count}.jpg'), bw_frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
            count -= 1

        cv2.imshow("Image", img) 

            



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])