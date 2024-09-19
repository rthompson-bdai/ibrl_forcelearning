
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


import json
from os import listdir
from os.path import isfile, join
import pickle

def jsonify(folder, out_name):
    subdirs = sorted([join(folder, f) for f in listdir(folder)])
    subdir_names = sorted([f for f in listdir(folder)])
    success_rates = {}
    for subdir, name in zip(subdirs, subdir_names): 
        successes = []
        if subdir[-4:] == "json":
            continue
        if len(listdir(subdir)) == 0:
            continue
        
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates[name] = success_rate#.append(success_rate)

    experiment_dict = success_rates#{"runs": subdir_names, "success_rates": success_rates}
    json.dump(experiment_dict, open(out_name, 'w'))


def run_jsonify(folder, out_name):
    subdirs = sorted([join(folder, f) for f in listdir(folder)])
    subdir_names = sorted([f for f in listdir(folder)])
    success_rates = {}
    factor_vals = {}
    for subdir, name in zip(subdirs, subdir_names): 
        if subdir[-4:] == "json":
            continue
        successes = []
        factors = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            successes.append(reward['rewards'][-1])
            
            factor_data = {}
            for key in reward["factors"].keys():
                factor_data[key] = []
                for j in range(len(reward["factors"][key])):
                    if type(reward["factors"][key][j])  == np.ndarray:
                        factor_data[key] += reward["factors"][key][j].astype(float).tolist()
                    elif type(reward["factors"][key][j])  == np.float32:
                        factor_data[key] += [reward["factors"][key][j].astype(float)]
                    else:
                        factor_data[key] += [reward["factors"][key][j]]
            factors.append(factor_data)
        success_rates[name] = successes
        factor_vals[name] = factors

    experiment_dict = {"successes": success_rates, "factors": factor_vals}
    json.dump(experiment_dict, open(out_name, 'w'))

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
    if not isfile(join(no_force_folder, "reward_data.json")):
        jsonify(no_force_folder,  join(no_force_folder, "reward_data.json"))
    success_rates = json.load(open(join(no_force_folder, "reward_data.json"), 'r'))
    if not isfile(join(force_folder, "reward_data.json")):
        jsonify(force_folder, join(force_folder, "reward_data.json"))
    force_success_rates = json.load(open(join(force_folder, "reward_data.json"), 'r'))

    fig = go.Figure()
    # subdirs = sorted([join(no_force_folder, f) for f in listdir(no_force_folder)])
    # subdir_names = sorted([f for f in listdir(no_force_folder)])
    # success_rates = {}
    # for subdir, subdir_name in zip(subdirs, subdir_names): 
    #     successes = []
    #     if len(listdir(subdir)) == 0:
    #         continue
    #     files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
    #     for f in files:
    #         reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
    #         successes.append(reward[-1])
    #     success_rate = sum(successes)/len(successes)
    #     success_rates[subdir_name] = success_rate

        

    # print(success_rates)

    # force_subdirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    # force_subdir_names = sorted([f for f in listdir(force_folder)])
    # force_success_rates = {}
    # for subdir, subdir_name in zip(force_subdirs, force_subdir_names): 
    #     successes = []
    #     if len(listdir(subdir)) == 0:
    #         continue
    #     files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
    #     for f in files:
    #         print(join(subdir, f))
    #         reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
    #         successes.append(reward[-1])
    #     success_rate = sum(successes)/len(successes)
    #     force_success_rates[subdir_name] = success_rate# - success_rates[subdir_name] 

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


def plot_by_factor(folders, leader):
    no_force = ['blue',  'aqua', 'aquamarine', 'darkturquoise', 'blue']
    force = ['red',  'peachpuff', 'palevioletred', 'lightpink', 'red']
    traces = {}
    for folder, colors, name in zip(folders,(force, no_force), ("force", "no_force")):

        seed_dirs = sorted([join(folder, f) for f in listdir(folder)])[3:4]
        seeds = sorted([f for f in listdir(folder)])[3:4]
        seed_factor_values = {}
        seed_factor_colors = {}

        i = 0
        for seed_dir, seed in zip(seed_dirs, seeds):
            if not isfile(join(seed_dir, "run_data.json")):
                run_jsonify(seed_dir, join(seed_dir, "run_data.json"))
            run_data = json.load(open(join(seed_dir, "run_data.json"), 'r'))
            factors = run_data['factors']
            successes = run_data['successes']
            for key in factors.keys():
                factor = get_factor(key)
                for factor_val, success in zip(factors[key], successes[key]):
                    #if success == 1:
                    if factor in seed_factor_values.keys():
                        seed_factor_values[factor].append(np.hstack(factor_val[factor]))
                        seed_factor_colors[factor].append(colors[i])
                        seed_factor_values["object_pos"].append(np.hstack(factor_val["object_pos"]))
                        seed_factor_colors["object_pos"].append(colors[i])
                    else:
                        seed_factor_values[factor] = [np.hstack(factor_val[factor])]
                        seed_factor_colors[factor] = [colors[i]]
                        seed_factor_values["object_pos"] = [np.hstack(factor_val["object_pos"])]
                        seed_factor_colors["object_pos"] = [colors[i]]
                if factor == 'light':
                    break
                
    
            i += 1
        print(len([factor_val[0] for factor_val in seed_factor_values[f'light'] if (factor_val[0] > .4 and factor_val[0] < .45) ]))
        # print(seed_factor_values.keys())
        # exit(0)
        for factor in seed_factor_values.keys():
            try:
                if factor not in traces.keys():
                    traces[factor] = {}
               
                for i in range(len(seed_factor_values[factor][0])):
                    if i not in traces[factor].keys():
                        traces[factor][i] = [go.Histogram(x=[factor_val[i] for factor_val in seed_factor_values[factor]], marker=dict(color=seed_factor_colors[factor]), name=f"{factor}_{i}_{name}")]
                    else:
                        traces[factor][i] += [go.Histogram(x=[factor_val[i] for factor_val in seed_factor_values[factor]], marker=dict(color=seed_factor_colors[factor]), name=f"{factor}_{i}_{name}")]
            except IndexError:
                continue

    for factor in traces.keys():
        fig = make_subplots(rows = len(traces[factor]), cols=1)
        for i in range(len(traces[factor])):
            for trace in traces[factor][i]:
                fig.add_trace(trace, row=i+1, col=1)
                # if factor == 'light':
                #     print(fig.data)
                #     f = fig.full_figure_for_development(warn=False)
                #     xbins = f.data[0].xbins
                #     plotbins = list(np.arange(start=xbins['start'], stop=xbins['end']+xbins['size'], step=xbins['size']))
                #     counts, bins = np.histogram(list(f.data[0].x), bins=plotbins)
                #     print(counts, bins)
                

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        fig.write_image(f"{leader}_{factor}_values.png")


def graph_success_from_folder(force_folder, no_force_folder, filename, factor=False, env=False):
    fig = go.Figure()
    if not isfile(join(no_force_folder, "reward_data.json")):
        jsonify(no_force_folder, "reward_data.json")
    success_rates = json.load(open(join(no_force_folder, "reward_data.json"), 'r'))
    if not isfile(join(force_folder, "reward_data.json")):
        jsonify(force_folder, "reward_data.json")
    force_success_rates = json.load(open(join(force_folder, "reward_data.json"), 'r'))
    # subdirs = sorted([join(no_force_folder, f) for f in listdir(no_force_folder)])
    # subdir_names = sorted([f for f in listdir(no_force_folder)])
    # success_rates = {}
    # for subdir, subdir_name in zip(subdirs, subdir_names): 
    #     successes = []
    #     if len(listdir(subdir)) == 0:
    #         continue
    #     files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
    #     for f in files:
    #         reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
    #         successes.append(reward[-1])
    #     success_rate = sum(successes)/len(successes)
    #     success_rates[subdir_name] = success_rate

    # force_subdirs = sorted([join(force_folder, f) for f in listdir(force_folder)])
    # force_subdir_names = sorted([f for f in listdir(force_folder)])
    # force_success_rates = {}
    # for subdir, subdir_name in zip(force_subdirs, force_subdir_names): 
    #     successes = []
    #     if len(listdir(subdir)) == 0:
    #         continue
    #     files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
    #     for f in files:
    #         reward = pickle.load(open(join(subdir, f), 'rb'))['rewards']
    #         successes.append(reward[-1])
    #     success_rate = sum(successes)/len(successes)
    #     force_success_rates[subdir_name] = success_rate# - success_rates[subdir_name] 

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

def get_seed_data_from_folder(folder, name, factor=False, env=False, color=None):
    seed_dirs = sorted([join(folder, f) for f in listdir(folder)])
    seeds = sorted([f for f in listdir(folder)])
    seed_success_rates = {}
    for seed_dir, seed in zip(seed_dirs, seeds):

        if not isfile(join(seed_dir, "reward_data.json")):
            jsonify(seed_dir, join(seed_dir, "reward_data.json"))
        success_rates = json.load(open(join(seed_dir, "reward_data.json"), 'r'))
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
            print(subkey)
            print(list(seed_data[seed][subkey].values()))
            if subkey not in all_data.keys():
                all_data[subkey] = [np.mean(list(seed_data[seed][subkey].values()))]
            else:
                all_data[subkey] += [np.mean(list(seed_data[seed][subkey].values()))]

    factors = list(all_data.keys())

    return go.Bar(x=factors, 
                y = [np.mean(all_data[factor]) for factor in factors],
                error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=[np.std(all_data[factor])/np.sqrt(len(all_data[factor])) for factor in factors],
                visible=True), name=name,
                marker=dict(color=color)
                )

def graph_success_from_all_folder(data_folders, seeded_data_folders, names, seeded_names, filename, factor=False, env=False):
    fig = go.Figure()

    all_data = []
    for folder in data_folders:
        if not isfile(join(folder, "reward_data.json")):
            jsonify(folder, join(folder, "reward_data.json"))
        success_rates = json.load(open(join(folder, "reward_data.json"), 'r'))
        all_data.append(success_rates)

    if factor:
        success_rates = [sort_data_by_factor(data) for data in all_data]
    elif env: 
        success_rates = [sort_data_by_env(data) for data in all_data]

    traces = []
    colors = ["cornflowerblue", "salmon"]
    for data, name, color in zip(success_rates, names, colors):
        key_list = list(data.keys())
        traces.append(go.Bar(x=key_list, y=[np.mean(list(data[k].values())) for k in key_list], name=name, marker=dict(color=color)))

    colors = ["cornflowerblue", "salmon", "blue", "red"]
    for folder, name, color in zip(seeded_data_folders, seeded_names, colors):
        traces += [get_seed_data_from_folder(folder, name, factor=factor, env=env, color=color)]

    for trace in traces:
        fig.add_trace(trace)
    
    fig.update_layout(barmode='group', yaxis_title="Success Rate", yaxis_range=[0,1], width=1000, height=600)
    fig.write_image(filename)



if __name__ == "__main__":
    plot_by_factor(['norm_warmup_no_prop_force_train/metaworld', 'rl_eval_train_no_prop_no_force/metaworld'], "noprop_eval_graphs/noprop_train_factors")

    #plot_by_factor('big_rl_eval_test/metaworld', 'big_rl_eval_test_no_force/metaworld', 'rl_100_test')
    # plot_by_factor('big_rl_eval_train/metaworld', 'big_rl_eval_train_no_force/metaworld', 'rl_100_train')



    # graph_success_deltas_over_seeds('rl_eval_train_no_prop_force/metaworld', 'rl_eval_test_no_prop_no_force/metaworld', 'rl_50_seed_factor_error.png', factor=True)
    # graph_success_deltas_over_seeds('rl_eval_test_no_prop_force/metaworld', 'rl_eval_test_no_prop_no_force/metaworld', 'rl_50_seed_env_error.png', env=True)
    # graph_success_from_all_folder(['bc_eval_test/metaworld/', 'no_prop_bc_eval_test/metaworld/'], 
    #                             #'big_rl_eval_train_no_force/metaworld'],
    #                             [], \
    #                             ["Images", "Images + force"],
    #                             [],
    #                             'bc_success_factor_test.png', factor=True)
    # graph_success_from_all_folder(['bc_eval_test/metaworld', 'no_prop_bc_eval_test/metaworld'], 
    #                             #'big_rl_eval_train_no_force/metaworld'],
    #                             [], \
    #                             ["Images", "Images + force"],
    #                             [],
    #                             'bc_success_env_test.png', env=True)
    # graph_success_from_all_folder(['bc_eval_train/metaworld/2024', 'no_prop_bc_eval_train/metaworld/2024'], 
    #                             #'big_rl_eval_train_no_force/metaworld'],
    #                             ['rl_eval_train_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_train/metaworld/', 
    #                                  'warmup_rl_eval_train_no_force/metaworld',
    #                                  'warmup_rl_eval_train/metaworld',], \
    #                             ["BC Images", "BC Images + force", ],
    #                             ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                             'all_all_success_env_train.png', env=True)
    # graph_success_from_all_folder(['bc_eval_test/metaworld/', 'no_prop_bc_eval_test/metaworld/'], 
    #                         #'big_rl_eval_train_no_force/metaworld'],
    #                         ['rl_eval_test_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_test/metaworld/', 
    #                                  'warmup_rl_eval_test_no_force/metaworld',
    #                                  'warmup_rl_eval_test/metaworld',], \
    #                         ["BC Images", "BC Images + force", ],
    #                         ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                         'all_all_success_env_test.png', env=True)
    # graph_success_from_all_folder(['bc_eval_train/metaworld/2024', 'no_prop_bc_eval_train/metaworld/2024'], 
    #                         #'big_rl_eval_train_no_force/metaworld'],
    #                         ['rl_eval_train_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_train/metaworld/', 
    #                                  'warmup_rl_eval_train_no_force/metaworld',
    #                                  'warmup_rl_eval_train/metaworld',], \
    #                         ["BC Images", "BC Images + force", ],
    #                         ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                         'all_all_success_factor_train.png', factor=True)
    # graph_success_from_all_folder(['bc_eval_test/metaworld/', 'no_prop_bc_eval_test/metaworld/'], 
    #                         #'big_rl_eval_train_no_force/metaworld'],
    #                         ['rl_eval_test_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_test/metaworld/', 
    #                                  'warmup_rl_eval_test_no_force/metaworld',
    #                                  'warmup_rl_eval_test/metaworld',], \
    #                         ["BC Images", "BC Images + force", ],
    #                         ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                         'all_all_success_factor_test.png', factor=True)















    # graph_success_from_all_folder(['bc_eval_train/metaworld/2024', 'no_prop_bc_eval_train/metaworld/2024'], 
    #                             #'big_rl_eval_train_no_force/metaworld'],
    #                             ['rl_eval_train_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_train/metaworld/', 
    #                                  'warmup_rl_eval_train_no_force/metaworld',
    #                                  'warmup_rl_eval_train/metaworld',], \
    #                             ["BC Images", "BC Images + force", ],
    #                             ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                             'all_all_success_env_train.png', env=True)
    # graph_success_from_all_folder(['bc_eval_test/metaworld/', 'no_prop_bc_eval_test/metaworld/'], 
    #                         #'big_rl_eval_train_no_force/metaworld'],
    #                         ['rl_eval_test_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_test/metaworld/', 
    #                                  'warmup_rl_eval_test_no_force/metaworld',
    #                                  'warmup_rl_eval_test/metaworld',], \
    #                         ["BC Images", "BC Images + force", ],
    #                         ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                         'all_all_success_env_test.png', env=True)
    # graph_success_from_all_folder(['bc_eval_train/metaworld/2024', 'no_prop_bc_eval_train/metaworld/2024'], 
    #                         #'big_rl_eval_train_no_force/metaworld'],
    #                         ['rl_eval_train_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_train/metaworld/', 
    #                                  'warmup_rl_eval_train_no_force/metaworld',
    #                                  'warmup_rl_eval_train/metaworld',], \
    #                         ["BC Images", "BC Images + force", ],
    #                         ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                         'all_all_success_factor_train.png', factor=True)
    # graph_success_from_all_folder(['bc_eval_test/metaworld/', 'no_prop_bc_eval_test/metaworld/'], 
    #                         #'big_rl_eval_train_no_force/metaworld'],
    #                         ['rl_eval_test_no_prop_no_force/metaworld/',
    #                                  'norm_warmup_no_prop_force_test/metaworld/', 
    #                                  'warmup_rl_eval_test_no_force/metaworld',
    #                                  'warmup_rl_eval_test/metaworld',], \
    #                         ["BC Images", "BC Images + force", ],
    #                         ["IBRL Images", "IBRL Images + Force", "IBRL Images + Prop",  "IBRL Images + Prop + Force"],
    #                         'all_all_success_factor_test.png', factor=True)
    # graph_success_from_all_folder(['bc_eval_train/metaworld', 'no_prop_bc_eval_train/metaworld'], 
    #                             #'big_rl_eval_train_no_force/metaworld'],
    #                             [], \
    #                             ["Images", "Images + force"],
    #                             [],
    #                             'bc_success_env.png', env=True)

    # graph_success_from_all_folder(['big_rl_eval_train/metaworld', \
    #                             'big_rl_eval_train_no_force/metaworld'],
    #                             ['rl_eval_train_no_prop_force/metaworld/', \
    #                             'rl_eval_train_no_prop_no_force/metaworld/'], \
    #                             ["Prop + force", "Prop only"],
    #                             ["Force only", "None"],
    #                             'rl_success_env_2026.png', env=True)


    # graph_success_from_all_folder([],
    #                                 ['warmup_rl_eval_train/metaworld',
    #                                 'warmup_rl_eval_train_no_force/metaworld',
    #                                 'norm_warmup_no_prop_force_train/metaworld/', 
    #                                 'rl_eval_train_no_prop_no_force/metaworld/'], 
    #                                 [],
    #                                 ["Force + Proprio", "Proprio Only", "Force only", "None"],
    #                                 'warm_rl_success_factor.png', factor=True)
    
    # graph_success_from_all_folder([],
    #                                 ['warmup_rl_eval_train/metaworld',
    #                                 'warmup_rl_eval_train_no_force/metaworld',
    #                                 'norm_warmup_no_prop_force_train/metaworld/', 
    #                                 'rl_eval_train_no_prop_no_force/metaworld/'], 
    #                                 [],
    #                                 ["Force + Proprio", "Proprio Only", "Force only", "None"],
    #                                 'warm_rl_success_env.png', env=True)
    
    # graph_success_from_all_folder([],
    #                                 ['warmup_rl_eval_test/metaworld',
    #                                 'warmup_rl_eval_test_no_force/metaworld',
    #                                 'norm_warmup_no_prop_force_test/metaworld/', 
    #                                 'rl_eval_test_no_prop_no_force/metaworld/'], 
    #                                 [],
    #                                 ["Force + Proprio", "Proprio Only", "Force only", "None"],
    #                                 'warm_rl_success_factor_test.png', factor=True)
    
    # graph_success_from_all_folder([],
    #                                 ['warmup_rl_eval_test/metaworld',
    #                                 'warmup_rl_eval_test_no_force/metaworld',
    #                                 'norm_warmup_no_prop_force_test/metaworld/', 
    #                                 'rl_eval_test_no_prop_no_force/metaworld/'], 
    #                                 [],
    #                                 ["Force + Proprio", "Proprio Only", "Force only", "None"],
    #                                 'warm_rl_success_env_test.png', env=True)

    
    # graph_success_deltas('warmup_rl_eval_test/metaworld/2025', 'warmup_rl_eval_test_no_force/metaworld/2025', 'with_prop_2025/rl_50_test_deltas.png')
    # graph_success_deltas('warmup_rl_eval_test/metaworld/2025', 'warmup_rl_eval_test_no_force/metaworld/2025', 'with_prop_2025/rl_50_test_factors_deltas.png', factor=True)
    # graph_success_deltas('warmup_rl_eval_test/metaworld/2025', 'warmup_rl_eval_test_no_force/metaworld/2025', 'with_prop_2025/rl_50_test_env_deltas.png', env=True)

    # graph_success_deltas('warmup_rl_eval_train/metaworld/2025', 'warmup_rl_eval_train_no_force/metaworld/2025', 'with_prop_2025/rl_50_train_deltas.png')
    # graph_success_deltas('warmup_rl_eval_train/metaworld/2025', 'warmup_rl_eval_train_no_force/metaworld/2025', 'with_prop_2025/rl_50_train_factors_deltas.png', factor=True)
    # graph_success_deltas('warmup_rl_eval_train/metaworld/2025', 'warmup_rl_eval_train_no_force/metaworld/2025', 'with_prop_2025/rl_50_train_env_deltas.png', env=True)

    # #graph_success_from_folder('warmup_rl_eval_test/metaworld/2025', 'warmup_rl_eval_test_no_force/metaworld/2025', 'with_prop_2025/rl_50_test_factor_mean.png', factor=True)
    # graph_success_from_folder('warmup_rl_eval_train/metaworld/2025', 'warmup_rl_eval_train_no_force/metaworld/2025', 'with_prop_2025/rl_50_train_factor_mean.png', factor=True)

    # #graph_success_from_folder('warmup_rl_eval_test/metaworld/2025', 'warmup_rl_eval_test_no_force/metaworld/2025', 'with_prop_2025/rl_50_test_env_mean.png', env=True)
    # graph_success_from_folder('warmup_rl_eval_train/metaworld/2025', 'warmup_rl_eval_train_no_force/metaworld/2025', 'with_prop_2025/rl_50_train_env_mean.png', env=True)
    
    