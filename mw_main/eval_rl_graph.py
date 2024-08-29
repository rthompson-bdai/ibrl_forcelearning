
import plotly
import plotly.graph_objects as go

from os import listdir
from os.path import isfile, join
import pickle

import numpy as np



def get_env(filename):
    return filename.split('_')[0]

def get_factor(filename):
    return filename[filename.find('_') + 1:]


def sort_data_by_factor(results_dict):
    by_factor = {}
    subdir_names = results_dict.keys()
    for name in subdir_names:
        factor = get_factor(name)
        if factor in by_factor.keys():
            by_factor[factor][name] = results_dict[name]
        else:
            by_factor[factor] = {name: results_dict[name]}
    return by_factor

def sort_data_by_env(results_dict):
    by_factor = {}
    subdir_names = results_dict.keys()
    for name in subdir_names:
        factor = get_env(name)
        if factor in by_factor.keys():
            by_factor[factor][name] = results_dict[name]
        else:
            by_factor[factor] = {name: results_dict[name]}
    return by_factor


def graph_factor_deltas(factor_dict):
    fig = go.Figure()
    factors = list(factor_dict.keys())
    fig.add_trace(go.Bar(x=factors, y = [np.mean(list(factor_dict[factor].values())) for factor in factors]))

    fig.write_image("rl_10_test_factor_deltas.png")


def graph_env_deltas(factor_dict):
    fig = go.Figure()
    factors = list(factor_dict.keys())
    fig.add_trace(go.Bar(x=factors, y = [np.mean(list(factor_dict[factor].values())) for factor in factors]))

    fig.write_image("rl_10_test_env_deltas.png")



def graph_success_deltas(force_folder, no_force_folder):

    fig = go.Figure()
    subdirs = sorted([join(no_force_folder, f) for f in listdir(no_force_folder)])
    subdir_names = sorted([f for f in listdir(no_force_folder)])
    success_rates = {}
    for subdir, subdir_name in zip(subdirs, subdir_names): 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates[subdir_name] = success_rate

    print(success_rates)

    force_subdirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    force_subdir_names = sorted([f for f in listdir(force_folder)])
    for subdir, subdir_name in zip(force_subdirs, force_subdir_names): 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        try:
            success_rates[subdir_name] = success_rate - success_rates[subdir_name] 
        except KeyError:
            continue
    print(success_rates)

    graph_factor_deltas(sort_data_by_factor(success_rates))

    # fig.add_trace(go.Bar(x=force_subdir_names, y=[success_rates[name] for name in force_subdir_names]))
    # fig.write_image("rl_10_train_deltas.png")



#plot delta based on factor
#get the factor dict 
#get the value 
#see if the values are the same values
#


def graph_success_from_folder(folder1, folder2):
    fig = go.Figure()
    subdirs = [join(folder1, f) for f in listdir(folder1)]
    subdir_names = [f for f in listdir(folder1)]
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

    fig.add_trace(go.Bar(x=subdir_names, y=success_rates, name="Train Factors"))

    subdirs = [join(folder2, f) for f in listdir(folder2)]
    subdir_names = [f for f in listdir(folder2)]
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

    fig.add_trace(go.Bar(x=subdir_names, y=success_rates, name="Test Factors"))
    fig.update_layout(barmode='group')
    fig.write_image("rl_test_successses.png")


if __name__ == "__main__":
    graph_success_deltas('rl_eval_test/metaworld', 'rl_eval_test_no_force/metaworld')