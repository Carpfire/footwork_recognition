import os
import torch 
import pickle as pkl
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np



class GRUNet(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layers, num_classes, dropout, preprocess=False):
        super().__init__()
        if preprocess:
            #TODO
            self.fc1 = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, 2* emb_dim),)
            emb_dim = 2*emb_dim
        
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=layers, batch_first=True,
        bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(), 
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(p=dropout), 
            nn.Linear(hidden_dim*2, num_classes)


        )

    def forward(self, x_pack):
        x, x_len = x_pack
        packed_x = nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        encoded_x = self.gru(packed_x)[0]
        unpacked_encoded_x, lens = nn.utils.rnn.pad_packed_sequence(encoded_x, batch_first=True)
        out = F.max_pool1d(unpacked_encoded_x.permute(0, 2, 1), unpacked_encoded_x.shape[1]).squeeze(2)
        decoded_x = self.fc(out)
        
        return F.softmax(decoded_x)

