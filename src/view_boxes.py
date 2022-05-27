import cv2
import os 
import numpy as np 
import json
import pickle as pkl
import time


def view_boxes(video, left_boxes, right_boxes, box_mode):
    cap = cv2.VideoCapture(video)
    i = 0
    with open(left_boxes, 'r') as read_file:
        l_boxes = json.load(read_file)
        read_file.close()
    with open(right_boxes, 'r') as read_file:
        r_boxes = json.load(read_file)
        read_file.close()
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while cap.isOpened() and pos < 70:
        success, img = cap.read()
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if box_mode == 'xyxy':
            lx1, ly1, lx2, ly2 = l_boxes[i][1]
            rx1, ry1, rx2, ry2 = r_boxes[i][1]
            cv2.rectangle(img, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 0, 0), 2)
            cv2.rectangle(img, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 2)
        else:
            lx, ly, lw, lh = l_boxes[i][1]
            rx, ry, rw, rh = r_boxes[i][1]
            cv2.rectangle(img, (int(lx), int(ly)), (int(lx+lw), int(ly+lh)),(255, 0, 0), 2)
            cv2.rectangle(img, (int(rx), int(ry)), (int(rx+rw), int(ry+rh)),(0, 255, 0), 2)
        cv2.imshow("Main Display", img)
        cv2.waitKey(1)
        time.sleep(.1)
        i += 1
def view_pose(video, left, right):
    with open(left, 'rb') as f:
        left_data = pkl.load(f)
        f.close()
    with open(right, 'rb') as f:
        right_data = pkl.load(f)
        f.close()
    cap = cv2.VideoCapture(video)
    for i in range(len(left_data)):
        succ, img = cap.read()
        left_pose = left_data[i][1][0]
        right_pose = right_data[i][1][0]
        print(left_pose)
        [[cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), thickness=1, lineType=cv2.FILLED) for x, y in zip(pose[0::2], pose[1::2])] for pose in (left_pose, right_pose)]
        cv2.imshow('poses',img)
        cv2.waitKey(1)

def main(title):

    view_boxes(f'Bern_2021\clips\\{title}.avi', f'Bern_2021\\2d_pose\{title}\\left\\boxes.json', f'Bern_2021\\2d_pose\{title}\\right\\boxes.json', 'xywh')


if __name__ == "__main__":
    #view_pose('Bern_2021\clips\BARDENET_FRA_vs_CANNONE_FRA__1.avi','Bern_2021\\vpd_res\BARDENET_FRA_vs_CANNONE_FRA__1__left.emb.pkl', 'Bern_2021\\vpd_res\BARDENET_FRA_vs_CANNONE_FRA__1__right.emb.pkl' )
    for name in os.listdir(os.path.join('Bern_2021', '2d_pose'))[1100:]:
        main(name)