import os 
import sys 

def main(parent_dir, child_dir, out_dir):
    dir_in = os.path.join(parent_dir, child_dir)
    dir_list = os.listdir(dir_in)
    dir_out = os.path.join(parent_dir, out_dir)
    key = '-converted'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for vid in dir_list:
        vid_in = os.path.join(dir_in, vid)
        if key in vid:
            vid = vid.replace(key, '')
            vid_out = os.path.join(dir_out, vid)
            # os.replace(vid_in,vid_out)
           
        else:
            continue

if __name__ == '__main__':
    main('C:\\Users\\liamc\\Desktop\\fencing_vision\\Bern_2021', 'clips', 'c_clips')