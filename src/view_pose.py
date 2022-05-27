from re import U
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import pickle as pkl
from itertools import chain
import torch
from vpd_fencing.models.module import FCNet, FCResNet, FCResNetPoseDecoder
import argparse



coco_bones = [[0, 2], [0,1], [1, 2], [2, 4], [4, 6], [1,3], [3, 5], [2, 8], [3, 7], [7, 8],[8, 10], [10, 12], [7, 9], [9, 11] ]

def view_poses(pose_file, decoder_path = None):
    if decoder_path: 
        decoder = FCNet(26, [128, 128], 2*26, dropout = 0)
        decoder.load_state_dict(torch.load(decoder_path))
        decoder.to('cuda:0')
        decoder.eval()
    with open(pose_file, 'rb') as f:
        data = pkl.load(f)
        f.close()
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    circles = ax.scatter([], [], s=4)
    lines = [ax.plot([],[], c='r', lw=2)[0] for bones in coco_bones]
    
        
    def pose_data_gen():
        for pose in data:
            if decoder_path: 
                pose = torch.Tensor(pose[1][0]).to('cuda:0')
                pose = decoder(pose).detach().cpu().numpy()
            else:
                pose = pose[1]
                
            yield np.c_[pose[0:26:2], -pose[1:26:2]]

    def vpd_data_gen():
        for pose in data:
            if decoder_path:
                print(pose[1][0, :].shape)
                pose = torch.Tensor(pose[1][0]).to('cuda:0')
                pose = decoder(pose).detach().cpu().numpy()
            else:
                pose = pose[1][0]
            yield np.c_[pose[0:26:2], -pose[1:26:2]]

    def draw_fencer(data):
        circles.set_offsets(data)

        return circles

    shape = data[0][1].shape
    print(f'Data Shape: {shape}')
    if  shape[0] == 26:
        data_gen = pose_data_gen
    if shape[0] == 2:
        data_gen = vpd_data_gen

    ani = FuncAnimation(fig, draw_fencer, data_gen)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_file', type=str)
    parser.add_argument('--decoder_path', type = str)

    args = parser.parse_args()
    view_poses(args.pose_file,  decoder_path =args.decoder_path)  

    
