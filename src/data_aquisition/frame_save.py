import cv2
import numpy as np
import sys as sys
import os as os



def display_menu(frame):
    #Menu Options and key presses
    font = cv2.FONT_HERSHEY_SIMPLEX
    coord_x, coord_y = (6, 9)
    fontScale = .4
    fontColor = (0,0,256)
    thickness = 1
    lineType = 2
    menu_options = ["left click: x, y coordinates","right click: BGR values", "s: Select Region"]
    offset = 0
    for option in menu_options:
        
        frame = cv2.putText(frame,option, 
        (coord_x, coord_y+offset), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        offset += 11
    
    return frame

            
def click_event(event, x, y, flags, params):
    '''
    Left Click to get the x, y coordinates.
    Right Click to get BGR color scheme at that position.
    '''
    frame = params
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices = crop(frame, x, y)
        rect =vertices
    elif event == cv2.EVENT_RBUTTONDOWN:
        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        print(f"B: {b}, G: {g}, R: {r} ")
    elif event == cv2.EVENT_MBUTTONDOWN:
        hsv_frame = hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv_frame[y, x, 0]
        s = hsv_frame[y, x, 1]
        v = hsv_frame[y, x, 2]
        print(f"H: {h}, S: {s}, V:{v}") 

def crop(frame, x, y):
    desired_height, desired_width = 28, 28
    vertices = (x-desired_height//2, y - desired_width//2), (x+desired_height//2, y + desired_height//2)
    cv2.rectangle(frame,vertices[0], vertices[1], (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    print(vertices)
    return vertices


                                                    
    
    


    

#Run Method for video exploration 
def run(video):
    # multipurpose tool for analyzing video data 
    rect = (597, 518, 625, 546)
    cap = cv2.VideoCapture(video)
    count = 0
    while cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = cap.read()

        frame = cv2.resize(frame, (1071, 604), interpolation = cv2.INTER_LANCZOS4) #Increase Size 
        key = cv2.waitKey(1) & 0xFF
        cv2.setMouseCallback("frame", click_event, frame)
        cv2.imshow("frame", frame)

        if key == ord('p'):
            cv2.imshow("frame",display_menu(frame)) #display menu
           
           # cv2.waitKey(0)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                #selectROI returns top left coordinate and then bottom right coordinate as 4-tuple
                rect = cv2.selectROI("frame", frame, showCrosshair=False, fromCenter=False)
                print(rect) # Display Coordinated

            if key == ord('x'):
                print(rect)
                bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f'r{count}.jpg', bw_frame[rect[1]:rect[3], rect[0]:rect[2]])
                count += 1


        if key == ord('f'):
            frame_count += 100
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(sys.argv[1])
    