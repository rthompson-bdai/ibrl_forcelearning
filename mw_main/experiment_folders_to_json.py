import json
from os import listdir
from os.path import isfile, join
import pickle

def jsonify(folder, out_name):
    subdirs = [join(folder, f) for f in listdir(folder)]
    subdir_names = [f for f in listdir(folder)]
    success_rates = []
    for subdir in subdirs: 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates.append(success_rate)

    experiment_dict = {"runs": subdir_names, "success_rates": success_rates}
    json.dump(experiment_dict, open(out_name, 'w'))


if __name__ == "__main__":
    jsonify('rl_eval_test/metaworld', 'rl_10_test_force.json')