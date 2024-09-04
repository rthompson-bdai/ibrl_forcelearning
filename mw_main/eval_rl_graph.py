
import plotly
import plotly.graph_objects as go

from os import listdir
from os.path import isfile, join
import pickle

import numpy as np

from plotly.subplots import make_subplots

def get_env(filename):
    return filename.split('_')[0]

def get_factor(filename):
    return filename[filename.find('_') + 1:]

def sort_data_by_factor(results_dict):
    by_factor = {}
    subdir_names = results_dict.keys()
    for name in subdir_names:
        factor = get_factor(name)
        # if get_env(name) in ['handle-pull', 'lever-pull', 'pick-place']:
        #     continue

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
        # if factor in ['handle-pull', 'lever-pull', 'pick-place']:
        #     continue
        # print(factor)
        if factor in by_factor.keys():
            by_factor[factor][name] = results_dict[name]
        else:
            by_factor[factor] = {name: results_dict[name]}
    return by_factor

def graph_factor_deltas(factor_dict, filename):
    fig = go.Figure()
    factors = list(factor_dict.keys())
    fig.add_trace(go.Bar(x=factors, y = [np.mean(list(factor_dict[factor].values())) for factor in factors]))
    fig.update_layout(yaxis_title="Force Performance Delta")
    fig.update_yaxes(range=[-.2, .2])

    fig.write_image(filename)


def graph_env_deltas(factor_dict, filename):
    fig = go.Figure()
    factors = list(factor_dict.keys())
    fig.add_trace(go.Bar(x=factors, y = [np.mean(list(factor_dict[factor].values())) for factor in factors]))
    fig.update_layout(yaxis_title="Force Performance Delta")
    fig.update_yaxes(range=[-.2, .2])

    fig.write_image(filename)

def graph_success_deltas_over_seeds(force_folder, no_force_folder, filename, factor=False, env=False):

    fig = go.Figure()
    seed_dirs = sorted([join(no_force_folder, f) for f in listdir(no_force_folder)])
    seeds = sorted([f for f in listdir(no_force_folder)])
    seed_success_rates = {}
    for seed_dir, seed in zip(seed_dirs, seeds):
        subdirs = sorted([join(seed_dir, f) for f in listdir(seed_dir)])
        subdir_names = sorted([f for f in listdir(seed_dir)])
        success_rates = {}
        for subdir, subdir_name in zip(subdirs, subdir_names): 
            successes = []
            if len(listdir(subdir)) == 0:
                continue
            files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
            for f in files:
                reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
                successes.append(reward[-1])
            success_rate = sum(successes)/len(successes)
            success_rates[subdir_name] = success_rate
        seed_success_rates[seed] = success_rates


    fig = go.Figure()
    seed_dirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    seeds = sorted([f for f in listdir(no_force_folder)])
    force_seed_success_rates = {}
    for seed_dir, seed in zip(seed_dirs, seeds):
        force_subdirs = sorted([join(seed_dir, f) for f in listdir(seed_dir)])
        force_subdir_names = sorted([f for f in listdir(seed_dir)])
        force_success_rates = {}
        for subdir, subdir_name in zip(force_subdirs, force_subdir_names): 
            successes = []
            if len(listdir(subdir)) == 0:
                continue
            files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
            for f in files:
                reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
                successes.append(reward[-1])
            success_rate = sum(successes)/len(successes)
            force_success_rates[subdir_name] = success_rate# - success_rates[subdir_name] 
        force_seed_success_rates[seed] = force_success_rates

    seed_data = {}
    for seed in seed_success_rates.keys():
        success_rates = seed_success_rates[seed]
        force_success_rates = force_seed_success_rates[seed]
        deltas = {}
        delta_keys  =[]
        for subdir in success_rates.keys():
            if subdir in force_success_rates.keys():
                delta_keys.append(subdir)
                deltas[subdir] = force_success_rates[subdir] - success_rates[subdir]
            
        #print(success_rates)
        if factor: 
            seed_data[seed] = sort_data_by_factor(deltas)
        elif env:
            seed_data[seed] = sort_data_by_env(deltas)
        else:
            seed_data[seed] = deltas

    all_data = {}
    for seed in seed_data.keys():
        for subkey in seed_data[seed]:
            if subkey not in all_data.keys():
                all_data[subkey] = [np.mean(list(seed_data[seed][subkey].values()))]
            else:
                all_data[subkey] += [np.mean(list(seed_data[seed][subkey].values()))]

    factors = list(all_data.keys())

    fig.add_trace(go.Bar(x=factors, 
                y = [np.mean(all_data[factor]) for factor in factors],
                error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=[np.std(all_data[factor])/np.sqrt(len(all_data[factor])) for factor in factors],
                visible=True)
                ))
    

    fig.update_layout(yaxis_title="Force Performance Delta")
    fig.update_yaxes(range=[-.2, .2])
            # fig.add_trace(go.Bar(x=delta_keys, y=[deltas[name] for name in delta_keys]))
            # fig.update_layout(yaxis_title="Force Performance Delta")
            # fig.update_yaxes(range=[-1, 1])
    fig.write_image(filename)


def graph_success_deltas(force_folder, no_force_folder, filename, factor=False, env=False):

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
            reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates[subdir_name] = success_rate

    print(success_rates)

    force_subdirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    force_subdir_names = sorted([f for f in listdir(force_folder)])
    force_success_rates = {}
    for subdir, subdir_name in zip(force_subdirs, force_subdir_names): 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        force_success_rates[subdir_name] = success_rate# - success_rates[subdir_name] 

    deltas = {}
    delta_keys  =[]
    for subdir in success_rates.keys():
        if subdir in force_success_rates.keys():
            delta_keys.append(subdir)
            deltas[subdir] = force_success_rates[subdir] - success_rates[subdir]
            
    #print(success_rates)
    if factor: 
        graph_factor_deltas(sort_data_by_factor(deltas), filename)
    elif env:
        graph_env_deltas(sort_data_by_env(deltas), filename)
    else:
        fig.add_trace(go.Bar(x=delta_keys, y=[deltas[name] for name in delta_keys]))
        fig.update_layout(yaxis_title="Force Performance Delta")
        fig.update_yaxes(range=[-1, 1])
        fig.write_image(filename)


def plot_by_factor(force_folder, no_force_folder, leader):
    subdirs = sorted([join(no_force_folder, f) for f in listdir(no_force_folder)])
    subdir_names = sorted([f for f in listdir(no_force_folder)])
    no_force_factors_values_dict = {}
    for subdir, subdir_name in zip(subdirs, subdir_names): 
        factor = get_factor(subdir_name)
        no_force_factors_values_dict[factor] = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            if reward['rewards'][-1] > 0: 
                no_force_factors_values_dict[factor].append(np.hstack(reward['factors'][factor]))

    subdirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    subdir_names = sorted([f for f in listdir(force_folder)])
    force_factors_values_dict = {}
    for subdir, subdir_name in zip(subdirs, subdir_names): 
        factor = get_factor(subdir_name)
        force_factors_values_dict[factor] = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            if reward['rewards'][-1] > 0: 
                force_factors_values_dict[factor].append(np.hstack(reward['factors'][factor]))

    for factor in force_factors_values_dict.keys():
        no_force_factors = no_force_factors_values_dict[factor]
        force_factors = force_factors_values_dict[factor]
        try:
            fig = make_subplots(len(force_factors[0]), 1)
            for i in range(len(force_factors[0])):
                fig.add_trace(go.Histogram(x=[factor_val[i] for factor_val in force_factors], name=f"{factor}_force_{i}"), row=i+1,col=1)
                fig.add_trace(go.Histogram(x=[factor_val[i] for factor_val in no_force_factors], name=f"{factor}_no_force_{i}"),row=i+1,col=1)
                # Overlay both histograms
                fig.update_layout(barmode='overlay')
                # Reduce opacity to see both histograms
                fig.update_traces(opacity=0.5)
                fig.write_image(f"{leader}_{factor}_values.png")
        except IndexError:
            continue


def graph_success_from_folder(force_folder, no_force_folder, filename, factor=False, env=False):
    
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
            reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates[subdir_name] = success_rate

    force_subdirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    force_subdir_names = sorted([f for f in listdir(force_folder)])
    force_success_rates = {}
    for subdir, subdir_name in zip(force_subdirs, force_subdir_names): 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        force_success_rates[subdir_name] = success_rate# - success_rates[subdir_name] 

    if factor:
        success_rates, force_success_rates = sort_data_by_factor(success_rates), sort_data_by_factor(force_success_rates)
        print(force_success_rates)
    elif env: 
        success_rates, force_success_rates = sort_data_by_env(success_rates), sort_data_by_env(force_success_rates)

    key_list = list(success_rates.keys())
    force_key_list = list(force_success_rates.keys())
    fig.add_trace(go.Bar(x=force_key_list, y=[np.mean(list(force_success_rates[k].values())) for k in key_list], name="Force"))
    fig.add_trace(go.Bar(x=key_list, y=[np.mean(list(success_rates[k].values())) for k in key_list], name="No Force"))
    
    fig.update_layout(barmode='group', yaxis_title="Success Rate", yaxis_range=[0,1])
    fig.write_image(filename)


def get_seed_data_from_folder(folder, name, factor=False, env=False):
    seed_dirs = sorted([join(folder, f) for f in listdir(folder)])
    seeds = sorted([f for f in listdir(folder)])
    seed_success_rates = {}
    for seed_dir, seed in zip(seed_dirs, seeds):
        subdirs = sorted([join(seed_dir, f) for f in listdir(seed_dir)])
        subdir_names = sorted([f for f in listdir(seed_dir)])
        success_rates = {}
        for subdir, subdir_name in zip(subdirs, subdir_names): 
            successes = []
            if len(listdir(subdir)) == 0:
                continue
            files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
            for f in files:
                reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
                successes.append(reward[-1])
            success_rate = sum(successes)/len(successes)
            success_rates[subdir_name] = success_rate
        seed_success_rates[seed] = success_rates

    seed_data = {}
    for seed in seed_success_rates.keys():
        success_rates = seed_success_rates[seed]  
        if factor: 
            seed_data[seed] = sort_data_by_factor(success_rates)
        elif env:
            seed_data[seed] = sort_data_by_env(success_rates)
        else:
            seed_data[seed] = success_rates

    all_data = {}
    for seed in seed_data.keys():
        for subkey in seed_data[seed]:
            if subkey not in all_data.keys():
                all_data[subkey] = [np.mean(list(seed_data[seed][subkey].values()))]
            else:
                all_data[subkey] += [np.mean(list(seed_data[seed][subkey].values()))]

    factors = list(all_data.keys())
    print([np.std(all_data[factor])/np.sqrt(len(all_data[factor])) for factor in factors])

    return go.Bar(x=factors, 
                y = [np.mean(all_data[factor]) for factor in factors],
                error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=[np.std(all_data[factor])/np.sqrt(len(all_data[factor])) for factor in factors],
                visible=True), name=name
                )



def graph_success_from_all_folder(data_folders, seeded_data_folders, names, seeded_names, filename, factor=False, env=False):
    fig = go.Figure()

    all_data = []
    for folder in data_folders: 
        subdirs = sorted([join(folder, f) for f in listdir(folder)])
        subdir_names = sorted([f for f in listdir(folder)])

        success_rates = {}
        for subdir, subdir_name in zip(subdirs, subdir_names): 
            successes = []
            if len(listdir(subdir)) == 0:
                continue
            files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
            for f in files:
                try:
                    reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
                except TypeError:
                    reward = pickle.load(open(join(subdir, f), 'rb'))
                successes.append(reward[-1])
            success_rate = sum(successes)/len(successes)
            success_rates[subdir_name] = success_rate
        all_data.append(success_rates)

    if factor:
        success_rates = [sort_data_by_factor(data) for data in all_data]
    elif env: 
        success_rates = [sort_data_by_env(data) for data in all_data]

    traces = []

    for data, name in zip(success_rates, names):
        key_list = list(data.keys())
        traces.append(go.Bar(x=key_list, y=[np.mean(list(data[k].values())) for k in key_list], name=name))

    for folder, name in zip(seeded_data_folders, seeded_names):
        traces += [get_seed_data_from_folder(folder, name, factor=factor, env=env)]

    for trace in traces:
        fig.add_trace(trace)
    
    fig.update_layout(barmode='group', yaxis_title="Success Rate", yaxis_range=[0,1])
    fig.write_image(filename)



if __name__ == "__main__":
    # plot_by_factor('big_rl_eval_test/metaworld', 'big_rl_eval_test_no_force/metaworld', 'rl_100_test')
    # plot_by_factor('big_rl_eval_train/metaworld', 'big_rl_eval_train_no_force/metaworld', 'rl_100_train')

    # graph_success_deltas_over_seeds('rl_eval_train_no_prop_force/metaworld', 'rl_eval_test_no_prop_no_force/metaworld', 'rl_50_seed_factor_error.png', factor=True)
    # graph_success_deltas_over_seeds('rl_eval_test_no_prop_force/metaworld', 'rl_eval_test_no_prop_no_force/metaworld', 'rl_50_seed_env_error.png', env=True)
    graph_success_from_all_folder(['bc_eval_train/metaworld', 'no_prop_bc_eval_train/metaworld'], 
                                #'big_rl_eval_train_no_force/metaworld'],
                                [], \
                                ["Images", "Images + force"],
                                [],
                                'bc_success_factor.png', factor=True)
    graph_success_from_all_folder(['bc_eval_train/metaworld', 'no_prop_bc_eval_train/metaworld'], 
                                #'big_rl_eval_train_no_force/metaworld'],
                                [], \
                                ["Images", "Images + force"],
                                [],
                                'bc_success_env.png', env=True)

    # graph_success_from_all_folder(['big_rl_eval_train/metaworld', \
    #                             'big_rl_eval_train_no_force/metaworld'],
    #                             ['rl_eval_train_no_prop_force/metaworld/', \
    #                             'rl_eval_train_no_prop_no_force/metaworld/'], \
    #                             ["Prop + force", "Prop only"],
    #                             ["Force only", "None"],
    #                             'rl_success_env_2026.png', env=True)


#    graph_success_from_all_folder(['big_rl_eval_train/metaworld', \
#                                 'big_rl_eval_train_no_force/metaworld'],
#                                 ['rl_eval_train_no_prop_force/metaworld/', \
#                                 'rl_eval_train_no_prop_no_force/metaworld/'], \
#                                 ["Prop + force", "Prop only"],
#                                 ["Force only", "None"],
#                                 'rl_success_factor_2026.png', factor=True)

#     graph_success_from_all_folder(['big_rl_eval_train/metaworld', \
#                                 'big_rl_eval_train_no_force/metaworld'],
#                                 ['rl_eval_train_no_prop_force/metaworld/', \
#                                 'rl_eval_train_no_prop_no_force/metaworld/'], \
#                                 ["Prop + force", "Prop only"],
#                                 ["Force only", "None"],
#                                 'rl_success_env_2026.png', env=True)



    
    # graph_success_deltas('rl_eval_test_no_prop_force/metaworld/2026', 'rl_eval_test_no_prop_no_force/metaworld/2026', '2026/rl_50_test_deltas.png')
    # graph_success_deltas('rl_eval_test_no_prop_force/metaworld/2026', 'rl_eval_test_no_prop_no_force/metaworld/2026', '2026/rl_50_test_factors_deltas.png', factor=True)
    # graph_success_deltas('rl_eval_test_no_prop_force/metaworld/2026', 'rl_eval_test_no_prop_no_force/metaworld/2026', '2026/rl_50_test_env_deltas.png', env=True)

    # graph_success_deltas('rl_eval_train_no_prop_force/metaworld/2026', 'rl_eval_train_no_prop_no_force/metaworld/2026', '2026/rl_50_train_deltas.png')
    # graph_success_deltas('rl_eval_train_no_prop_force/metaworld/2026', 'rl_eval_train_no_prop_no_force/metaworld/2026', '2026/rl_50_train_factors_deltas.png', factor=True)
    # graph_success_deltas('rl_eval_train_no_prop_force/metaworld/2026', 'rl_eval_train_no_prop_no_force/metaworld/2026', '2026/rl_50_train_env_deltas.png', env=True)

    # graph_success_from_folder('rl_eval_test_no_prop_force/metaworld/2026', 'rl_eval_test_no_prop_no_force/metaworld/2026', '2026/rl_50_test_factor_mean.png', factor=True)
    # graph_success_from_folder('rl_eval_train_no_prop_force/metaworld/2026', 'rl_eval_train_no_prop_no_force/metaworld/2026', '2026/rl_50_train_factor_mean.png', factor=True)

    # graph_success_from_folder('rl_eval_test_no_prop_force/metaworld/2026', 'rl_eval_test_no_prop_no_force/metaworld/2026', '2026/rl_50_test_env_mean.png', env=True)
    # graph_success_from_folder('rl_eval_train_no_prop_force/metaworld/2026', 'rl_eval_train_no_prop_no_force/metaworld/2026', '2026/rl_50_train_env_mean.png', env=True)
    
    