import os
import shutil
from argparse import ArgumentParser

def rename_and_copy(file):
    basename, suffix = file.split('.')
    left_name = ''.join([basename, '_left', '.',suffix])
    right_name = ''.join([basename, '_right', '.',suffix])
    shutil.copy(file, left_name)
    os.rename(file, right_name)

def main(dir):
    dir_list = os.listdir(dir)
    for file in dir_list:
        rename_and_copy(os.path.join(dir,file))
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    main(args.dir)
    