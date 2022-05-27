import numpy as np
import os 
import random

def main():
    #randomly select 10% of dataset for testing, move to seperate directories
    path = 'Bern_2021'
    crop_path = os.path.join(path, 'crops')
    pose_path = os.path.join(path, '2d_pose')
    emb_path = os.path.join(path, 'embs')

    crop_tpath = os.path.join(path, 't_crops')
    pose_tpath = os.path.join(path, 't_2d_pose')
    emb_tpath = os.path.join(path, 't_embs')

    feat_paths = [crop_path, pose_path, emb_path]
    test_paths = [crop_tpath, pose_tpath, emb_tpath]

    for p in test_paths:
        if not os.path.exists(p):
            os.mkdir(p)
    pose_names = os.listdir(pose_path)
    dsize = len(pose_names)
    test_size = int(.1 * dsize)
    names = random.sample(pose_names,test_size)
    for f_path, t_path in zip(feat_paths, test_paths):
        for name in names:
            in_path = os.path.join(f_path, name)
            out_path = os.path.join(t_path, name)
            if "embs" in f_path:
                in_path_l, in_path_r  = '', ''
                in_path_l = in_path_l.join([in_path, '_left_.emb.pkl'])
                in_path_r = in_path_r.join([in_path, '_right_.emb.pkl'])
                out_path_l, out_path_r = '', ''
                out_path_l = out_path_l.join([out_path, '_left_.emb.pkl'])
                out_path_r = out_path_r.join([out_path, '_right_.emb.pkl'])
                os.replace(in_path_l, out_path_l)
                os.replace(in_path_r, out_path_r)

            else:

                os.replace(in_path, out_path)
            


if __name__ == '__main__':
    main()