import os 
from random import sample
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("file", type=str,)
    return parser.parse_args()

def load_all(file):
    with open(file, 'r') as f:
        data = f.readlines()
    f.close()
    return data

def val_create(file):
    data = load_all(file)
    split = int(len(data)*.1)
    val_data = sample(data, split)
    val_data = [''.join([val.rsplit(' ',3)[0], '\n']) for val in val_data]
    with open('val.ids.txt', 'w') as f:
        for line in val_data:
            f.write(line)
    f.close()

if __name__ == '__main__':
    val_create(get_args().file)
