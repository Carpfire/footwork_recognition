# Creation of Boxes.json for VPD

import profile
from detectron2.config import get_cfg
from detectron2.structures.boxes import pairwise_iou, Boxes
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from queue import Queue
from tqdm import tqdm
import numpy as np
import gzip
import json
import cv2
import os
import torch
from multiprocessing import Pool
from line_profiler import LineProfiler



# Preprocess Bounding Boxes, ie fencers not referees or spectators
pose_checkpnt = '.\pose_estimation\litehrnet30_coco_384x288-a3aef5c4_20210626.pth'
pose_config = '.\pose_estimation\\topdown_heatmap_litehrnet_30_coco_384x288.py'
pose_model = init_pose_model(pose_config, pose_checkpnt)
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

cfg.MODEL.WEIGHTS = ".\\vpd_fencing\\output\\referee_crowd_fencer.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
predictor = DefaultPredictor(cfg)


""" Uses trained detectron2 object detector to detect fencers. Uses max intersection over union of current detections
 againsts previous left or right fencer detection for continuity between frames. 
 TODO: extra detection for missed detections.  """


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return torch.Tensor([x1, y1, w, h])


def xywh_to_xyxy(box):
    x, y, w, h = box
    x2 = x + w
    y2 = y + h
    return torch.Tensor([x, y, x2, y2])

def get_area(box):
    x, y, w, h = box
    return w*h

def good_detection(l_box, r_box):
    #TODO: Deal with fencers not being totally in line
    #Currently uses the intersection of the two bounding boxes on the y-axis
    box_list = [l_box, r_box]
    box_list.sort(reverse=True, key=get_area)
    l_box, r_box = box_list
    lx1, ly1, lw, lh = l_box
    rx1, ry1, rw, rh = r_box
    ly2 = ly1 + lh
    ry2 = ry1 + rh
    max_y1 = max(ly1, ry1)
    min_y2 = min(ly2, ry2)
    overlap = min_y2 - max_y1
    if overlap < .3*lh:
        # print("Bad Detect")
        return False
    else:
        # print("Good Detect")
        return True
def bad_iou(iou):
    return True if len(torch.where(iou > .3)[0]) == 0 else False
    
# def best_boxes(boxes):
#     boxes.sort(reverse = True, key = get_area)
#     return boxes_2[:2]



# TODO: Fix issue with switching left, v. right labels on fencer when they cross 
# TODO: Better Solution to bad v good detections(fencers aren't always level from camera perspective)

def create_file(video, output, viz=True):
    if not os.path.exists(output):
        os.mkdir(output)
    cap = cv2.VideoCapture(video)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    end_frame = (total//2)
    fencer_left = []
    fencer_right = []
    pose_left = []
    pose_right = []
    tracker_left = cv2.TrackerCSRT_create()
    tracker_right = cv2.TrackerCSRT_create()
    left_buffer = Queue(maxsize=1)
    right_buffer = Queue(maxsize=1)
    img_buffer = Queue(maxsize=1)
    initial_detection = False
    begin_tracking = False
    position = 0
    while cap.isOpened() and position < end_frame:
        success, img = cap.read()
        #print(img.shape)
        if not success:

            print(f"Video {os.path.basename(video).replace('.avi', '')} failed to read")
            break
        else:
            position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            instances = predictor(img)["instances"]
            boxes_xyxy = instances[instances.pred_classes == 1].get_fields()['pred_boxes'].to("cpu").tensor #get fencer bounding boxes
            boxes = [xyxy_to_xywh(box) for box in boxes_xyxy]
            #print(boxes)
            if not initial_detection:
                # initial detection uses biggest bounding box to determine fencers
                if len(boxes) >= 2:
                    boxes.sort(reverse=True, key= get_area)
                    box1, box2 = boxes[:2]
#BLUE_CIMINI_ITA_vs_FARDZINOV_RUS__16
                    if box1[0] < box2[0]: 
                        l_box = box1
                        r_box = box2
                        initial_detection = True
                        left_buffer.put(l_box)
                        right_buffer.put(r_box)
                        img_buffer.put(img)

                    elif box2[0] < box1[0]:
                        r_box = box1
                        l_box = box2
                        initial_detection = True
                        left_buffer.put(l_box)
                        right_buffer.put(r_box)
                        img_buffer.put(img)
                else:
                    print("Poor Initial Detection")
                    
            else:

                left_prev = left_buffer.get()
                right_prev = right_buffer.get()
                prev_img = img_buffer.get()
                left_iou = pairwise_iou(Boxes(xywh_to_xyxy(left_prev).reshape(1, -1)), Boxes(boxes_xyxy))[0]
                right_iou = pairwise_iou(Boxes(xywh_to_xyxy(right_prev).reshape(1, -1)), Boxes(boxes_xyxy))[0]
                #Once a bounding box is lost revert to CSRT tracking with previous bounding boxes 
                
                # print(f"Left IOU {left_iou}")
                # print(f"Right IOU {right_iou}")
                try:
                    left_ind = int((left_iou == max(left_iou)).nonzero()[0,0])
                    right_ind = int((right_iou == max(right_iou)).nonzero()[0,0])

                except ValueError: 
                    print("Value Error: Defaulting to CSRT")
                    left_ind, right_ind = 1, 1


                if left_ind != right_ind and not (bad_iou(left_iou) or bad_iou(right_iou)):
                    l_box, r_box = boxes[left_ind], boxes[right_ind]
                    if begin_tracking:
                        begin_tracking = False
                        print("Tracking Stopped")
                    # if not good_detection(l_box.tensor.tolist()[0], r_box.tensor.tolist()[0]):
                    #     if not begin_tracking:
                    #         begin_tracking = True
                    #         print("Tracker Initiated")
                    #         l_init = tuple(int(el) for el in xyxy_to_xywh(left_prev.tensor.numpy().tolist()[0]))
                    #         r_init = tuple(int(el) for el in xyxy_to_xywh(right_prev.tensor.numpy().tolist()[0]))
                    #         tracker_left.init(prev_img, l_init)
                    #         tracker_right.init(prev_img, r_init)

                    #     if begin_tracking:
                    #         _, l_box = tracker_left.update(img)
                    #         _, r_box = tracker_right.update(img)
                else: 
                    if not begin_tracking:
                        begin_tracking = True
                        print("Tracker Initiated")
                        l_init = tuple(int(el) for el in left_prev)
                        r_init = tuple(int(el) for el in right_prev)
                        tracker_left.init(prev_img, l_init)
                        tracker_right.init(prev_img, r_init)

                    if begin_tracking:
                        _, l_box = tracker_left.update(img)
                        _, r_box = tracker_right.update(img)
                        r_box = torch.Tensor(r_box)
                        l_box = torch.Tensor(l_box)
                        
                # if type(l_box) == Boxes:    
                #     left_buffer.put(l_box)
                #     right_buffer.put(r_box)
                # elif type(l_box != Boxes):
                #     left_buffer.put(Boxes(np.array(xywh_to_xyxy(l_box)).reshape(1, -1)))
                #     right_buffer.put(Boxes(np.array(xywh_to_xyxy(r_box)).reshape(1, -1)))
                left_buffer.put(l_box)
                right_buffer.put(r_box)
                img_buffer.put(img)

            #Run Pose estimation on bounding box 
            
            # TODO more accurate fencer selection from bounding boxes
            if initial_detection:
                box_dict = [{'bbox':l_box.numpy()}, {'bbox':r_box.numpy()}]
                res = inference_top_down_pose_model(pose_model, img, box_dict, format='xywh')[0]
                left_pose, right_pose = res[0]['keypoints'], res[1]['keypoints']
                left_pose_score, right_pose_score = left_pose[:,-1].mean(), right_pose[:,-1].mean()
                if viz:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    # if type(l_box) == Boxes:

                    #     l_box = xyxy_to_xywh(l_box.tensor.numpy().tolist()[0])
                    #     r_box = xyxy_to_xywh(r_box.tensor.numpy().tolist()[0])
                    #     l_box = tuple(int(el) for el in l_box)
                    #     r_box = tuple(int(el) for el in r_box)
                        
                    cv2.rectangle(
                        img,
                        (int(l_box[0]), int(l_box[1])),
                        (int(l_box[0] + l_box[2]), int(l_box[1] + l_box[3])),
                        (255, 0, 0),
                        2,
                    )
                    cv2.rectangle(
                        img,
                        (int(r_box[0]), int(r_box[1])),
                        (int(r_box[0] + r_box[2]), int(r_box[1] + r_box[3])),
                        (0, 255, 0),
                        2,
                    )
                    [[cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), thickness=2, lineType=cv2.FILLED) for x, y in pose[:,:2]] for pose in (left_pose, right_pose)]
                    cv2.imshow("Main", img)
                    cv2.waitKey(1)
                    # if first_detection:
                    #     tracker_left.init(img, l_box)
                    #     tracker_right.init(img, r_box)
                    #     first_detection = False

                l_box, r_box = l_box.numpy().tolist(), r_box.numpy().tolist()
                    

                fencer_left.append([position, l_box])
                fencer_right.append([position, r_box])
                pose_left.append([position, [[float(left_pose_score), l_box, left_pose.tolist()]]])
                pose_right.append([position, [[float(right_pose_score), r_box, right_pose.tolist()]]])


        
             
    left_path = os.path.join(output, "left")
    right_path = os.path.join(output, "right")


    if not os.path.exists(left_path):
        os.mkdir(left_path)

    if not os.path.exists(right_path):
        os.mkdir(right_path)

    with open(os.path.join(left_path, "boxes.json"), "w") as file:
        json.dump(fencer_left, file)
        file.close()

    with open(os.path.join(right_path, "boxes.json"), "w") as file:
        json.dump(fencer_right, file)
        file.close()
    
    with gzip.open(os.path.join(left_path, "coco_keypoints.json.gz"), 'w') as file:
        file.write(json.dumps(pose_left).encode('utf-8'))

    with gzip.open(os.path.join(right_path, "coco_keypoints.json.gz"), 'w') as file:
        file.write(json.dumps(pose_right).encode('utf-8'))




def main(parent_dir):
    pose_dir = os.path.join(parent_dir, "2d_pose")
    if not os.path.exists(pose_dir):
        os.mkdir(pose_dir)
    dir_list = os.listdir(os.path.join(parent_dir, "clips"))

    index = dir_list.index('BLUE_DI_VEROLI_ITA_vs_SVICHKAR_UKR__11.avi')
    for vid in tqdm(dir_list[index:]):
        basename = os.path.basename(vid).replace(".avi", "")
        print(basename)
        sub_pose_dir = os.path.join(pose_dir, basename)
        if not os.path.exists(sub_pose_dir):
            os.mkdir(sub_pose_dir)
        # lp = LineProfiler()
        # lp_wrapper = lp(create_file)
        # lp_wrapper(os.path.join(parent_dir, "clips", vid), sub_pose_dir)
        # lp.print_stats()
        create_file(os.path.join(parent_dir, "clips", vid), sub_pose_dir)

def worker_func(args):
    return create_file(*args)


def parallel_main(clip_dir, out_dir, viz):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    video_names = [os.path.join(clip_dir,name) for name in os.listdir(clip_dir)]
    
    worker_args = [(v, os.path.join(out_dir, os.path.basename(v).rstrip('.avi')), viz) for v in video_names]

    if viz:
        parallelism = 1
    else:
        parallelism = 2
    with Pool(parallelism) as p, tqdm(total=len(video_names)) as pbar:
        for video_name in p.imap_unordered(worker_func, worker_args):
            pbar.set_description(video_name, refresh=True)
            pbar.update(1)
    print('Done')


if __name__ == "__main__":
    parallel_main("Bern_2021\clips", "Bern_2021\\2d_pose", False)

