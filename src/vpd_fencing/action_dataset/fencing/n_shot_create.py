import os 
from argparse import ArgumentParser
from random import sample

def get_args():
    parser = ArgumentParser()
    parser.add_argument('dir', type = str)
    parser.add_argument('n_shot', type=int)
    parser.add_argument('--include', type=list, default=[
          'advancing', 
          'retreating', 
          'fleche', 
          'lunge', 
          'half step forward', 
          'half step backward'])
    parser.add_argument('--exp_num', type=int)

    return parser.parse_args()

def n_shot_train(dir, n , actions, exp_num):
    with open(os.path.join(dir, 'all.txt'), 'r') as f:
        anns = f.readlines()
    f.close()
    annotation_dict = {label:[] for label in actions}
    for line in anns:
        base, action = line.split(' ', 1)
        action = action.replace('\n', '')
        if action in annotation_dict.keys():
            annotation_dict[action].append(base)
    print(list(len(value) for value in annotation_dict.values()))
    with open(os.path.join(dir, f'train_{n}_{exp_num}.ids.txt'), 'w') as f:
        for action_list in annotation_dict.values(): 
            actions = sample(action_list, n if len(action_list)>n else len(action_list))
            actions = [''.join([action, '\n']) for action in actions]
            f.writelines(actions)
    f.close()

if __name__ == '__main__':
    args = get_args()
    n_shot_train(args.dir, args.n_shot, args.include, args.exp_num)
    




        

