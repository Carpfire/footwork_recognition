import os
import random
from re import S
import torch 
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    label_list, data_list = [], []
    for (data, label) in batch:

        label_list.append(label)
        data_list.append(data.squeeze())
    
    data_len = [d.shape[0] for d in data_list]
    data_list = pad_sequence(data_list, batch_first=True, padding_value=0)
    return (data_list, torch.Tensor(data_len)), torch.Tensor(label_list)


def train_valid_split(annotation_file, n_shot, subset=None):
    if subset is None: 
        action_subset = {line.split(' ', 1)[1].replace('\n','') for line in open(annotation_file, 'r').readlines()}
    else: 
        action_subset = subset
    data_dict = {action:[] for action in action_subset} 
    #data_counter = {action:0 for action in action_subset}
    for line in open(annotation_file, 'r'):
        name, action = line.split(' ', 1)
        start_frame, end_frame = name.split(':')[-2:]
        action = action.replace('\n', '')
        
        if action in action_subset and start_frame < end_frame:
                #data_counter[action] += 1    
                data_dict[action].append(name)
    
    train_data = []
    validation_data = []
    for action, names in data_dict.items():
        names = np.array(names)
        sample_size = len(names)
        inds = np.array(random.sample(range(0, sample_size), n_shot if sample_size > n_shot else sample_size//2))
        train_data.extend([(name,action) for name in names[inds]])
        validation_data.extend([(name, action) for name in names[~inds]])

    
    #Handle Filtering, Subsetting and splitting of the data
    return train_data, validation_data

class PoseDataset(Dataset):
    def __init__(self, data, pose_dir, shape=52):
            #ids and n_shot redundant if using n_shot file
            #[(name:side:start_frame:end_frame, action_label)]
            self.shape = shape
            self.pose_dir = pose_dir
            self.action_labels = []
            self.names = []
            random.shuffle(data)
            for name, action in data:
                self.names.append(name)
                self.action_labels.append(action)
            
            self.label_dict = {lab:i for lab, i in zip(set(sorted(self.action_labels)), range(len(self.action_labels)))}



    def __len__(self):
        return len(self.action_labels)


    def __getitem__(self, idx):
        basename, side, start_frame, end_frame = self.names[idx].split(':')
        assert int(end_frame) > int(start_frame), f"{basename}, {side}, {start_frame}, {end_frame}"
        pose_path = os.path.join(self.pose_dir,''.join([basename, '__', side, '.emb.pkl']))
        with open(pose_path, 'rb') as f:
            pose_emb =  pkl.load(f)
        f.close()
        poses = torch.Tensor(np.array([tup[1].reshape(-1, self.shape) for tup in pose_emb[int(start_frame) +1:int(end_frame) + 1]]))
        label = self.label_dict[self.action_labels[idx]]
        
        return poses, label