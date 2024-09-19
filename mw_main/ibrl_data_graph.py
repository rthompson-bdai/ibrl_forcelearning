import plotly

from os import listdir
from os.path import isfile, join
import pickle

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import h5py
# rotation util code is from mujoco_worldgen/util/rotation.py
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))

def make_arrow(x, y, z, color):
    datum = [
    {
        'x': x,
        'y': y,
        'z': z,
        'mode': "lines",
        'type': "scatter3d",
        'line': {
            'color': color,
            'width': 3
        }
    },
    {
        "type": "cone",
        'x': [x[1]],
        'y': [y[1]],
        'z': [z[1]],
        'u': [0.3*(x[1]-x[0])],
        'v': [0.3*(y[1]-y[0])],
        'w': [0.3*(z[1]-z[0])],
        'anchor': "tip", # make cone tip be at endpoint
        'hoverinfo': "none",
        'colorscale': [[0, color], [1, color]], # color all cones blue
        'showscale': False,
    }]

    traces = [go.Line(**datum[0]), go.Cone(**datum[1])]
    return traces


def make_quiver(start_points, end_points, colors):
    arrows = []
    for start,end,color in zip(start_points, end_points, colors):
        arrows += make_arrow((start[0], end[0]),(start[1], end[1]),(start[2], end[2]),color)
    return arrows


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


import json
from os import listdir
from os.path import isfile, join
import pickle
import torch

#sigh
def arrayify(list_of_tensors):
    replacement = []
    for t in list_of_tensors:
        if type(t) == torch.Tensor:
            replacement.append(t.cpu().numpy())
        else:
            replacement.append(t)
    return np.array(replacement).tolist()


def jsonify(folder, out_name):
    subdirs = sorted([join(folder, f) for f in listdir(folder)])
    subdir_names = sorted([f for f in listdir(folder)])
    all_data = {'use_bcs': {}, "qas": {}, "both_actions":{}, "forces":{}}

    for subdir, name in zip(subdirs, subdir_names): 
        if subdir[-4:] == 'json':
            continue
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            data = pickle.load(open(join(subdir, f), 'rb'))
            if name in all_data['use_bcs'].keys():
                all_data['use_bcs'][name].append( arrayify(data['use_bcs']))
                all_data['qas'][name].append( arrayify(data['qas']))
                all_data['both_actions'][name].append( arrayify(data['both_actions']))
                all_data['forces'][name].append( arrayify(data['forces']))
            else:
                all_data['use_bcs'][name]  = [ arrayify(data['use_bcs'])]
                all_data['qas'][name]  = [ arrayify(data['qas'])]
                all_data['both_actions'][name]  = [ arrayify(data['both_actions'])]
                all_data['forces'][name]  = [ arrayify(data['forces'])]
    experiment_dict = all_data
    json.dump(experiment_dict, open(join(folder, out_name), 'w'))


def get_seed_data_from_folder(folder, name, factor=False, env=False):
    seed_dirs = sorted([join(folder, f) for f in listdir(folder)])
    seeds = sorted([f for f in listdir(folder)])
    seed_ibrl_data = {}
    for seed_dir, seed in zip(seed_dirs, seeds):
        if not isfile(join(seed_dir, "ibrl_data.json")):
            jsonify(seed_dir, "ibrl_data.json")
        ibrl_data = json.load(open(join(seed_dir, "ibrl_data.json"), 'r'))['use_bcs']
        seed_ibrl_data[seed] = ibrl_data

    seed_data = {}
    for seed in seed_ibrl_data.keys():
        success_rates = seed_ibrl_data[seed]  
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
                all_data[subkey] = [1 - np.mean(np.hstack(list(seed_data[seed][subkey].values())))]
            else:
                all_data[subkey] += [1 - np.mean(np.hstack(list(seed_data[seed][subkey].values())))]

    factors = list(all_data.keys())
    return go.Bar(x=factors, 
                y = [np.mean(all_data[factor]) for factor in factors],
                error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=[np.std(all_data[factor])/np.sqrt(len(all_data[factor])) for factor in factors],
                visible=True), name=name
                )
                
def graph_bc_counts_over_seed(force_folder, no_force_folder, filename):
    fig = go.Figure()
    traces = []
    traces.append(get_seed_data_from_folder(force_folder, "Force", factor=False, env=True))
    #traces.append(get_seed_data_from_folder(no_force_folder, "No Force", factor=False, env=True))
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(barmode='group', yaxis_title="Proportion of RL to whole episode", yaxis_range=[0,1])
    fig.write_image(filename)


def graph_bc_counts(force_folder, no_force_folder, filename):
    # if not isfile(join(no_force_folder, "ibrl_data.json")):
    #     jsonify(no_force_folder,  join(no_force_folder, "ibrl_data.json"))
    # success_ibrl = json.load(open(join(no_force_folder, "ibrl_data.json"), 'r'))['use_bc']

    if not isfile(join(force_folder, "ibrl_data.json")):
        jsonify(force_folder, join(force_folder, "ibrl_data.json"))
    force_success_ibrl = json.load(open(join(force_folder, "ibrl_data.json"), 'r'))['use_bcs']
    fig = go.Figure()

    #ibrl_data = sort_data_by_env(success_ibrl)
    force_ibrl_data = sort_data_by_env(force_success_ibrl)

    #ibrl_data_keys = list(ibrl_data.keys())
    force_ibrl_data_keys  = list(force_ibrl_data.keys())

    #print([np.mean(list(force_ibrl_data[k].values())) for k in force_ibrl_data_keys])
    fig.add_trace(go.Bar(x=force_ibrl_data_keys, y=[1 - np.mean(np.hstack(list(force_ibrl_data[k].values()))) for k in force_ibrl_data_keys], name="Force"))
    #fig.add_trace(go.Bar(x=ibrl_data_keys, y=[np.mean(list(ibrl_data[k].values())) for k in ibrl_data_keys], name="No Force"))
    
    fig.update_layout(barmode='group', yaxis_title="Proportion of RL to whole episode", yaxis_range=[0,1])
    fig.write_image(filename)

def graph_force_info(force_folder, no_force_folder, leader,):
    if not isfile(join(force_folder, "ibrl_data.json")):
        jsonify(force_folder, "ibrl_data.json")
    force_success_ibrl = json.load(open(join(force_folder, "ibrl_data.json"), 'r'))

    # if not isfile(join(no_force_folder, "ibrl_data.json")):
    #     jsonify(no_force_folder, "ibrl_data.json")
    # no_force_success_ibrl = json.load(open(join(force_folder, "ibrl_data.json"), 'r'))
    dirs = [join(no_force_folder, f) for f in listdir(no_force_folder)]
    dir_names = [f for f in listdir(no_force_folder)]
    no_force_hist_data = {}
    for dir, name in zip(dirs, dir_names):
        print(dir)
        f = h5py.File(join(dir,'dataset.hdf5'))
        env = name.split("_")[0]# + "_" + dir.split("_")[2]
        no_force_hist_data[env]=[]

        num_episode: int = len(list(f["data"].keys()))  # type: ignore
        # print(f"Raw Dataset size (#episode): {num_episode}")
        episode_lens = []
        for episode_id in range(num_episode):
            episode_tag = f"demo_{episode_id}"
            episode = f[f"data/{episode_tag}"]
            # print(episode.keys())
            # print(episode["obs"]["prop"])
            # exit(0)
            no_force_hist_data[env].append(episode["obs"]["prop"][:, -6:])

    print(no_force_hist_data.keys())

    force_ibrl_data = sort_data_by_env(force_success_ibrl['forces'])
    #no_force_ibrl_data = sort_data_by_env(no_force_success_ibrl['forces'])
    force_ibrl_data_keys  = list(force_ibrl_data.keys())
    print(force_ibrl_data_keys)

    
    #i = 0
    for key in force_ibrl_data_keys:
        fig = make_subplots(rows=len(force_ibrl_data_keys), cols=1,)
        env_data = force_ibrl_data[key]
        #nf_env_data = no_force_ibrl_data[key]
        x_labels = ["X", "Y", "Z", "R", "P", "Y"]
        for i in range(len(x_labels)):
            hist_data = []
            
            for subkey in env_data.keys():
                    hist_data.append(np.array(env_data[subkey][0])[:,i])
                    #no_force_hist_data.append(np.array(nf_env_data[subkey][0])[:,i])
            hist_data = np.concatenate(hist_data)
            nf_hist_data = np.concatenate(no_force_hist_data[key])
            nf_hist_data = nf_hist_data[:,i]
            fig.add_trace(go.Histogram(x=hist_data, name=f'Forces at Test Time', xbins={'size':.2}, marker={'color':"red"}, ), row=i+1,col=1)
            fig.add_trace(go.Histogram(x=nf_hist_data, name=f'Training Data Forces', xbins={'size':.2}, marker={'color':"blue"}, ), row=i+1,col=1)
            fig.update_xaxes(row=i+1, col=1, range=[-5, 5], tickfont = dict(size=7)) 
            fig.update_yaxes(title_text=x_labels[i], type="log",tickmode="linear", row=i+1, col=1, tickfont = dict(size=7)) 
        #i += 1

        fig.update_layout(barmode='overlay', title=f"Experienced Forces: {key}", showlegend=False, width=600, height=800)
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        print("DONE")
        fig.write_image(f'{leader}_{key}.png')


def graph_bc_counts_force_info(force_folder, no_force_folder, leader, env=False, factor=False, count=6):
    # maybe bin the number of times rl was selected by the associated force info
    if not isfile(join(force_folder, "ibrl_data.json")):
        jsonify(force_folder, "ibrl_data.json")
    force_success_ibrl = json.load(open(join(force_folder, "ibrl_data.json"), 'r'))

    if env: 
        use_bcs_all = sort_data_by_env(force_success_ibrl['use_bcs'])
        forces_all = sort_data_by_env(force_success_ibrl['forces'])
    elif factor: 
        use_bcs_all = sort_data_by_factor(force_success_ibrl['use_bcs'])
        forces_all = sort_data_by_factor(force_success_ibrl['forces'])

    for key in list(use_bcs_all.keys()):
        use_bcs = use_bcs_all[key]
        forces = forces_all[key]
        fig = make_subplots(rows=6, cols=1)
        x_labels = ["X", "Y", "Z", "R", "P", "Y"]
        for i in range(count):
            hist_data = []
            bc_hist_data = []
            for subkey in use_bcs.keys():
                    hist_data.append(np.array(forces[subkey][0])[np.hstack(use_bcs[subkey][0]) == 0, i],)
                    bc_hist_data.append(np.array(forces[subkey][0])[np.hstack(use_bcs[subkey][0]) == 1, i],)
            hist_data = np.concatenate(hist_data)
            bc_hist_data = np.concatenate(bc_hist_data)
            fig.add_trace(go.Histogram(x=hist_data, name=f'RL Action Count', xbins={'size':.2}, marker={'color':"red"}, ), row=i+1,col=1)
            fig.add_trace(go.Histogram(x=bc_hist_data, name=f'BC Action Count', xbins={'size':.2}, marker={'color':"blue"}, ), row=i+1,col=1)
            fig.update_xaxes(row=i+1, col=1, range=[-5, 5], tickfont = dict(size=7)) 
            fig.update_yaxes(title_text=x_labels[i], type="log",tickmode="linear", row=i+1, col=1, tickfont = dict(size=7)) 
        fig.update_layout(barmode='overlay', title=f"Action Counts by Force Value: {key}", showlegend=False, width=600, height=800)
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        fig.write_image(f'{leader}_{key}.png')
    

def graph_bc_counts_by_timestep(force_folder, no_force_folder, filename):
    #TODO: sort by successful/unsuccessful out of curiosity 
    if not isfile(join(force_folder, "ibrl_data.json")):
        jsonify(force_folder, "ibrl_data.json")
    force_success_ibrl = json.load(open(join(force_folder, "ibrl_data.json"), 'r'))['use_bcs']
    force_ibrl_data = sort_data_by_env(force_success_ibrl)
    force_ibrl_data_keys  = list(force_ibrl_data.keys())

    fig = make_subplots(rows=len(force_ibrl_data_keys), cols=1)
    i = 0
    for key in force_ibrl_data_keys:
        env_data = force_ibrl_data[key]
        hist_data = []
        for subkey in env_data.keys():
            for datum in env_data[subkey]:
                hist_data.append(np.nonzero(np.hstack(datum) == 0)[0])
        hist_data = np.concatenate(hist_data)
        fig.add_trace(go.Histogram(x=hist_data, name=f"{key}_force"), row=i+1,col=1)
        i += 1

    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    fig.write_image(filename)


def graph_policy(policy_file, filename):
    #TODO need to fix to add position info for graphing 
    data = pickle.load(open(policy_file, 'rb'))
    actions = np.array(data['actions'])
    both_actions = data['both_actions']
    use_bcs = data['use_bcs']

    print(actions)

    true_actions = [np.sum(actions[:i, :], axis=0)[:3] for i in range(len(actions))]

    BC_CHOSEN_COLOR = 'red'
    BC_REJECTED_COLOR = 'gray'
    RL_CHOSEN_COLOR = 'blue'
    RL_REJECTED_COLOR = 'gray'

    bc_chosen = []
    bc_rejected = []
    rl_chosen = []
    rl_rejected = []
    for true_action, both_action, use_bc in zip(true_actions, both_actions, use_bcs):
        both_action = both_action.squeeze()
        if use_bc == 1.0:
            rl_rejected.append(true_action + both_action[0][0, :3].cpu().numpy())
            bc_chosen.append(true_action + both_action[1, :3].cpu().numpy())
        else:
            rl_chosen.append(true_action + both_action[0, :3].cpu().numpy())
            bc_rejected.append(true_action + both_action[1, :3].cpu().numpy())

    bc_chosen_arrows = make_quiver(true_actions, bc_chosen, [BC_CHOSEN_COLOR for _ in bc_chosen])
    bc_rejected_arrows = make_quiver(true_actions, bc_rejected, [BC_REJECTED_COLOR for _ in bc_rejected])
    rl_chosen_arrows = make_quiver(true_actions, rl_chosen, [RL_CHOSEN_COLOR for _ in rl_chosen])
    rl_rejected_arrows = make_quiver(true_actions, rl_rejected, [RL_REJECTED_COLOR for _ in rl_rejected])

    all_traces = bc_chosen_arrows + bc_rejected_arrows + rl_chosen_arrows + rl_rejected_arrows

    fig = go.Figure()
    for trace in all_traces:
        fig.add_trace(trace)
    fig.write_image(filename)


if __name__ == "__main__":

    # graph_force_info("no_prop_bc_eval_train/metaworld/2024", "bc_data/metaworld", "aaa_force",)


    graph_bc_counts_force_info("norm_warmup_no_prop_force_train/metaworld/2024", "rl_eval_train_no_prop_no_force/metaworld", "ibrl_data/2024/force_hist_env", env=True)
    graph_bc_counts_force_info("norm_warmup_no_prop_force_train/metaworld/2024", "rl_eval_train_no_prop_no_force/metaworld", "ibrl_data/2024/force_hist_factor", factor=True)

    # graph_bc_counts_force_info("warmup_rl_eval_train/metaworld/2024", "warmup_rl_eval_train_no_force/metaworld", "ibrl_data/2024/prop_force_hist_env_", env=True)
    # graph_bc_counts_force_info("warmup_rl_eval_train/metaworld/2024", "warmup_rl_eval_train_no_force/metaworld", "ibrl_data/2024/prop_force_hist_factor_", factor=True)

    # graph_bc_counts_force_info("warmup_rl_eval_train_no_force/metaworld/2024", "warmup_rl_eval_train_no_force/metaworld", "ibrl_data/2024/prop_no_force_hist_env_", env=True, count=4)
    # graph_bc_counts_force_info("warmup_rl_eval_train_no_force/metaworld/2024", "warmup_rl_eval_train_no_force/metaworld", "ibrl_data/2024/prop_no_force_hist_factor_", factor=True, count=4)
    
    #graph_policy("norm_warmup_no_prop_force_train/metaworld/2024/button-press_arm_pos/episode0.mp4.reward.pkl", "ibrl_data/2024/button-press_arm_pos_1.png")


    #graph_bc_counts_by_timestep("norm_warmup_no_prop_force_train/metaworld/2024", "rl_eval_train_no_prop_no_force/metaworld", "ibrl_data/2024/timestep.png")
    #graph_bc_counts_over_seed("norm_warmup_no_prop_force_train/metaworld", "rl_eval_train_no_prop_no_force/metaworld", "ibrl_data/no_prop.png")
