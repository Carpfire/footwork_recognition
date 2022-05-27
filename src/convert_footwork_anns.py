import os 
import json 
from argparse import ArgumentParser

fps = 24

def load_json(file):
    with open(file, 'r') as f:
        annotations = json.load(f)
    f.close()
    return annotations
#Desired Format: video_name:side:frame_start:frame_end action
def parse_annotation(annotation): #list of dictionaries for specific video 
    title = os.path.basename(annotation['video_url']).strip('.mp4')
    name, side = title.rsplit('_', 1)
    name = name.split('-')[1]
    #iterate through tricks to get each action
    for action in annotation['tricks']:
        start_time, end_time, labels = action.values()
        #convert start, end time to frames
        start_frame, end_frame = int(start_time*fps), int(end_time*fps)
        yield name, side, start_frame if start_frame > 0 else 0, end_frame if end_frame < 130 else 130, labels[0]

def parse_file(file, out_dir):
    annotations = load_json(file)#return list of dictionaries
    out_path = os.path.join(out_dir, 'all.txt')
    with open(out_path, 'w') as f:
        for ann in annotations:
            for name, side, start_frame, end_frame, label in parse_annotation(ann):
                out_str = f'{name}:{side}:{start_frame}:{end_frame} {label}\n'
                f.write(out_str)
    f.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file_in', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()
    parse_file(args.file_in, args.out_dir)


