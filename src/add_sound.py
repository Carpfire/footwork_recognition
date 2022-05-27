import os
from argparse import ArgumentParser


#ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -i BARDENET_FRA_vs_CANNONE_FRA__10_left.mp4 -c:v copy -c:a aac -shortest output.mp4
def add_sound(file_in, file_out):
    os.system(f"""ffmpeg -f lavfi -i 
    anullsrc=channel_layout=stereo:sample_rate=44100 -y
    -i {file_in} -c:v 
    copy -c:a aac -shortest {file_out} """.replace('\n', ''))

def main(dir, out_dir):
    dir_list = os.listdir(dir)
    for file in dir_list:
        add_sound(os.path.join(dir, file), os.path.join(out_dir, file))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()
    main(args.dir, args.out_dir)