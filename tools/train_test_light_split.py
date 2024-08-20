

import numpy
import random
import json
import argparse
import string

def train_test_split(lighting_params, n_train):
    N=6
    exp_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
    train_filename = f'lighting_train_{n_train}_{exp_string}.json'
    test_filename = f'lighting_test_{len(lighting_params) - n_train}_{exp_string}.json'
    numpy.random.shuffle(lighting_params)
    train, test = lighting_params[:n_train], lighting_params[n_train:]

    with open(train_filename, 'w') as fout:
        json.dump(train, fout)

    with open(test_filename, 'w') as fout:
        json.dump(test, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--n_train", type=int, default=3)
    args = parser.parse_args()

    with open(args.file) as f:  
        params = json.load(f)

    train_test_split(params, args.n_train)



