import time, cv2, os, sys
import numpy as np
from keyclipwriter import KeyClipWriter
from queue import Queue

from torchvision import transforms
from joblib import load
''' This script has 4 functions eval score uses a Multiclass Logistic Regression model to evaluate the score given by the score box overlay in a fencing vision youtube video 
eval time does the same for the time. save_touch goes through a video and cuts and saves clips every time a touch is scored (light goes off and score changes) and process files will take files 
in the precut directory, iterate throught them, and run save_touch'''




score_model = load('models\score_classifier.joblib')
seconds_model = load('models\seconds_classifier.joblib')
minutes_model = load('models\min_classifier.joblib')

tensor_normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#def of lower/upper bounds of overlay colours for HSV Red"
lower_red = np.array([170,100,100], dtype = "uint16")
upper_red = np.array([180,255,255], dtype = "uint16")

#def of lower/upper bounds of overlay colours for HSV Green"
lower_green = np.array([60,90,90], dtype = "uint16")
upper_green = np.array([80,255,255], dtype = "uint16")

last_score = (-1, -1)
bout_over = False

def process_files(dir):
	#Function for iterating through each video in directory then runninng touch save 
	path = os.path.join(dir,'precut')
	for video in os.listdir(path):

		name = video.removesuffix('.mp4')
		splits = name.split('_')
		splits.reverse()

		filename = []
		for chars in splits:
			if chars == 'WC' or chars == 'PODIUM':
				break 
			else: filename.append(chars)

		filename.reverse()
		title = ''.join([''.join([name, '_']) for name in filename])
		video = os.path.join(path, video)
		save_touch(video, dir, title)
	




def eval_score(frame, classifier):
	#Function for checking and returning score using trained Logistic Regressor 
	right_box = (597, 518, 625, 546)
	left_box = (447, 517, 475, 545)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	left_im = frame[left_box[1]:left_box[3], left_box[0]:left_box[2]]
	right_im = frame[right_box[1]:right_box[3], right_box[0]:right_box[2]]
	left_im = tensor_normalize(left_im)
	right_im = tensor_normalize(right_im)
	left_score = classifier.predict(left_im.reshape(1, -1))
	right_score = classifier.predict(right_im.reshape(1, -1))
	
	return left_score[0], right_score[0]

	

def eval_time(frame, m_classifier, s_classifier):
	minutes_box = (506, 520, 22, 28)
	seconds_box = (533, 519, 37, 28)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	minutes_im = frame[minutes_box[1]:minutes_box[1]+minutes_box[3], minutes_box[0]:minutes_box[0]+minutes_box[2]]
	seconds_im = frame[seconds_box[1]:seconds_box[1]+seconds_box[3], seconds_box[0]:seconds_box[0]+seconds_box[2]]
	minutes_im = tensor_normalize(minutes_im)
	seconds_im = tensor_normalize(seconds_im)
	minutes = m_classifier.predict(minutes_im.reshape(1, -1))
	seconds = s_classifier.predict(seconds_im.reshape(1, -1))

	return minutes[0], seconds[0]


def check_if_valid(video, position, last_score):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    success, img = cap.read()
    img = cv2.resize(img, (1071, 604), interpolation = cv2.INTER_LANCZOS4) #Increase Size 
    curr_score = eval_score(img, score_model)
    curr_time = eval_time(img, minutes_model, seconds_model)
    future_time = curr_time
    while curr_time == future_time:
        position += 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, img = cap.read()
        img = cv2.resize(img, (1071, 604), interpolation = cv2.INTER_LANCZOS4) #Increase Size 
        future_time = eval_time(img, minutes_model, seconds_model)
    
    future_lscore, future_rscore = eval_score(img, score_model)
    print(curr_score, (future_lscore, future_rscore), last_score)
    if future_lscore != curr_score[0] or future_rscore != curr_score[1]:
        if (future_lscore, future_rscore) != last_score:
            last_score = (future_lscore, future_rscore)
            return True, last_score
        else: return False, last_score
    else:
        return False, last_score


#Main Problem is it will sometimes not record the very last touch of the bout
def save_touch(video, sub_dir, main_title):

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = 60
    touch_number = 0
    track_frames = False
    right_light_on = False
    left_light_on = False
    valid_touch= False
    consec_frames = 0

    last_score = (-1, -1)

    base_dir = os.path.basename(sub_dir)
    clip_path = os.path.join(os.getcwd(), base_dir, 'clips')

    kcw = KeyClipWriter(bufSize=100)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not os.path.exists(clip_path):
        os.mkdir(clip_path)
    try:
        while cap.isOpened():

            success, img = cap.read()

            if not success:
                break
            
            else:
                img = cv2.resize(img, (1071, 604), interpolation = cv2.INTER_LANCZOS4) #Increase Size  
                position = cap.get(cv2.CAP_PROP_POS_FRAMES)

                #Conversion to HSV for masking 
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                #Select lights and mask	
                lights_overlay = hsv_img[545:545+13, 133:133+806]
                touch_left = cv2.inRange(lights_overlay, lower_red, upper_red).sum()
                touch_right = cv2.inRange(lights_overlay, lower_green, upper_green).sum()

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                
                #fast forward key 	
                if key == ord('f'):
                    position += 100
                    cap.set(cv2.CAP_PROP_POS_FRAMES, position)

                if touch_left > 500000:
                    if not kcw.recording:
                        left_light_on = True
                if touch_right > 500000:
                    if not kcw.recording:
                        right_light_on = True
            
                        #Check if valid touch
                if right_light_on and touch_right < 500000 and not kcw.recording:
                    right_light_on = False
                    valid_touch, last_score = check_if_valid(video, position, last_score)               
                elif left_light_on and touch_left < 500000 and not kcw.recording:
                    left_light_on = False
                    valid_touch, last_score = check_if_valid(video, position, last_score)

                if valid_touch: 
                    print("Valid Touch")
                    
                    valid_touch = False
                    touch_number += 1
                    track_frames = True
                    video_name = os.path.join(clip_path, f"{main_title}_{touch_number}.avi")
                    kcw.start(video_name, fourcc, 30)
                
                kcw.update(img)
                if kcw.recording and track_frames:
                    consec_frames += 1

                if consec_frames > max_frames and kcw.recording:
                    consec_frames=0
                    kcw.finish() 

                cv2.imshow("main_display", img)
                cv2.waitKey(1)

    except cv2.error: pass



if __name__ == "__main__":
    process_files(".\Bern_2021")
