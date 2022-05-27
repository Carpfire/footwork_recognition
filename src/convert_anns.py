import os
import numpy as np
import json
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=str)
    return parser.parse_args()

def load_json(file):
    with open(file, 'r') as f:
        annotations = json.load(f)
    f.close()
    return annotations

def convert_file(jfile):
    
    def convert_annotation(single_ann):
        print(single_ann)
        video_name = os.path.basename(single_ann['video']).split('-')[1].replace('.mp4', '')
        if isinstance(single_ann['choice'], dict):
            labels = [label.lower().split(' ', 1) for label in single_ann['choice']['choices']]
        else: 
            labels = [single_ann['choice'].lower().split(' ',1)]
        labels_dict = {side:action for side, action in labels}
        return video_name, labels_dict
        
    with open('all.txt', 'w') as f:   
        for ann in jfile:
            name, labels = convert_annotation(ann)
            for side, label in labels.items():
                out_str = f'{name}:{side} {label}\n'
                f.write(out_str)
    f.close()

def main():
    j_file = load_json(get_args().file)
    convert_file(j_file)

if __name__ == '__main__':
    main()



    

            


