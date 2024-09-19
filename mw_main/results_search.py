
from os import listdir
from os.path import isfile, join
import pickle



if __name__ == "__main__":
     #list two folders
     #check every pickle file in the folder
     #find anywhere the first folder succeeds but the second folder fails
     #try and generate the gifs already and then do this i guess 

    folder_a = 'rl_viz_force/metaworld'
    folder_b = 'bc_viz_no_force/metaworld'

    subdirs_a = [join(folder_a, f) for f in listdir(folder_a)]
    subdirs_b = [join(folder_b, f) for f in listdir(folder_b)]
    subdir_names = [f for f in listdir(folder_b)]

    for subdir_a, subdir_b, subdir_name in zip(subdirs_a, subdirs_b, subdir_names): 
        successes = []
        if len(listdir(subdir_a)) == 0 or len(listdir(subdir_b)) == 0 :
            continue
        files_a = [f for f in listdir(subdir_a) if (isfile(join(subdir_a, f)) and f[-3:] == 'pkl')]
        files_b = [f for f in listdir(subdir_b) if (isfile(join(subdir_b, f)) and f[-3:] == 'pkl')]
        for f_a, f_b in zip(files_a, files_b):
            reward_a = pickle.load(open(join(subdir_a, f_a), 'rb'))['rewards'][-1]
            reward_b = pickle.load(open(join(subdir_b, f_b), 'rb'))['rewards'][-1]
            if reward_a > reward_b:
                print(reward_a)
                print(reward_b)
                print(join(subdir_a, f_a))


