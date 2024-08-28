
import argparse
import json
import os
import random

import h5py
import numpy as np


from generate_metaworld_dataset import EnvironmentConfig, MainConfig
from train_rl_mw import BC_DATASETS
#from robomimic.envs.wrappers import ForceBinningWrapper

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image
import pyrallis
from dataclasses import dataclass, field
import dataclasses
from env.metaworld_wrapper import PixelMetaWorld



#import mimicgen_envs
# def choose_mimicgen_environment():
#     """
#     Prints out environment options, and returns the selected env_name choice

#     Returns:
#         str: Chosen environment name
#     """

#     # try to import robosuite task zoo to include those envs in the robosuite registry
#     try:
#         import robosuite_task_zoo
#     except ImportError:
#         pass

#     # all base robosuite environments (and maybe robosuite task zoo)
#     robosuite_envs = set(suite.ALL_ENVIRONMENTS)

#     # all environments including mimicgen environments
#     import mimicgen_envs
#     all_envs = set(suite.ALL_ENVIRONMENTS)

#     # get only mimicgen envs
#     only_mimicgen_envs = sorted(all_envs - robosuite_envs)

#     # keep only envs that correspond to the different reset distributions from the paper
#     envs = [x for x in only_mimicgen_envs if x[-1].isnumeric()]

#     # Select environment to run
#     print("Here is a list of environments in the suite:\n")

#     for k, env in enumerate(envs):
#         print("[{}] {}".format(k, env))
#     print()
#     try:
#         s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
#         # parse input into a number within range
#         k = min(max(int(s), 0), len(envs))
#     except:
#         k = 0
#         print("Input is not valid. Use {} by default.\n".format(envs[k]))

#     # Return the chosen environment name
#     return envs[k]


if __name__ == "__main__":
    import rich.traceback

    rich.traceback.install()
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    #run(cfg)
    cfg.env_cfg.factor_kwargs = {}
    env_kwargs = dataclasses.asdict(cfg.env_cfg)
    # env_kwargs['norm'] = True
    # env_kwargs['norm_dataset'] = BC_DATASETS[cfg.env_cfg.env_name]
    
    env = PixelMetaWorld(**env_kwargs)

    fig = make_subplots(rows=3, cols=1, subplot_titles=("EE Force", "Demo", "EE Torque"))

    forces = []
    torques = []
    images = []
    rl_obs, _ = env.reset()
    print(rl_obs)
    forces.append(np.array(rl_obs['prop'][-6:-3].detach().cpu()))
    torques.append(np.array(rl_obs['prop'][-3:].detach().cpu()))

    for _ in range(cfg.env_cfg.episode_length):
        heuristic_action = env.get_heuristic_action(clip_action=True)
        rl_obs, reward, terminal, _, _ = env.step(heuristic_action)
        forces.append(np.array(rl_obs['prop'][-6:-3].detach().cpu()))
        torques.append(np.array(rl_obs['prop'][-3:].detach().cpu()))

    # force  = states[:, 32:35]
    # torque = states[:, 35:38]

    # v_func_force = np.vectorize(force_binning)
    # v_func_torque = np.vectorize(torque_binning)

    #print(np.array(forces) - initial_force)
    forces=np.array(forces)#v_func_force(np.array(forces) - initial_force)
    torques=np.array(torques)#v_func_torque(np.array(torques))
    print(forces.shape)
    
    fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name='X force'), row=1, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name='Y force'), row=1, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name='Z force'), row=1, col=1)

    #print(dir(go))
    #figm = px.imshow(images, binary_string=True, facet_col=0, facet_col_wrap=10)
    #fig.add_trace(figm.data[0], 2, 1)
    #print(len(images))
    for i, image in enumerate(images):
        fig.add_trace( go.Image(z=np.flipud(image), x0=i*len(image),), row=2, col=1)

    fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name='R torque'), row=3, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name='P torque'), row=3, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name='Y torque'), row=3, col=1)

    # other_obs = f["data/{}/obs".format(ep)]
    # #print(other_obs['robot0_ee_force'])
    # other_forces = other_obs['robot0_ee_force']#[other_obs[i]["robot0_ee_force"] for i in range(len(other_obs))]
    # other_torques = other_obs['robot0_ee_torque']#[other_obs[i]["robot0_ee_torque"] for i in range(len(other_obs))]
    # fig.add_trace(go.Scatter(x = np.arange(len(other_forces)), y= other_forces[:,0], name='other X force'), row=1, col=1)
    # fig.add_trace(go.Scatter(x = np.arange(len(other_forces)), y= other_forces[:,1], name='other Y force'), row=1, col=1)
    # fig.add_trace(go.Scatter(x = np.arange(len(other_forces)), y= other_forces[:,2], name='other Z force'), row=1, col=1)

    # fig.add_trace(go.Scatter(x = np.arange(len(other_torques)), y= other_torques[:,0], name='other R torque'), row=3, col=1)
    # fig.add_trace(go.Scatter(x = np.arange(len(other_torques)), y= other_torques[:,1], name='other P torque'), row=3, col=1)
    # fig.add_trace(go.Scatter(x = np.arange(len(other_torques)), y= other_torques[:,2], name='other Y torque'), row=3, col=1)


    #fig.show()
    fig.write_image("fig1.png")

    #actions = np.array(f["data/{}/actions".format(ep)][()])
    exit(0)

