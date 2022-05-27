import pickle as pkl
import os 
import sys
#BARDENET_FRA_vs_CANNONE_FRA__1_left_.emb.pkl -> BARDENET_FRA_vs_CANNONE_FRA__1__right.emb.pkl
def rename(file, dir_path):
    split_string = file.rsplit('_', 2)
    out = os.path.join(dir_path, ''.join([split_string[0], '__', split_string[1], split_string[2]]))
    os.rename(os.path.join(dir_path, file), out)





if __name__ == '__main__':
    dir_path = sys.argv[1]
    for file in os.listdir(dir_path):
        rename(file, dir_path)