
import argparse
import json
import os
import random

import h5py
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config
#from robosuite.utils.input_utils import *

#from robomimic.envs.wrappers import ForceBinningWrapper

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image


import plotly.graph_objects as go

import plotly.io as pio
print(pio.renderers.default)# = "notebook"
#print(pio.renderers)


# For testing whether a number is close to zero
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


def graph_perturbations(cur_pos, mean_actions, perturbed_actions, forces=None, scores=None):
    if scores is not None:
       
        scores -= np.min(scores)
        scores = np.array(scores).reshape(1, -1)/np.max(scores) #* .95

        from plotly.express.colors import sample_colorscale
        arrows = []

        for p_action, score in zip(perturbed_actions, scores.T):
            #p_action= p_action.reshape(1, -1)
            #p_norm = np.linalg.norm(p_action[:, :3])
            #print(p_norm)

            c = sample_colorscale('Viridis', [score[0]])[0]
            arrows += make_arrow([cur_pos[0], cur_pos[0]+p_action[0]], 
                                     [cur_pos[1], cur_pos[1]+p_action[1]], 
                                     [cur_pos[2], cur_pos[2]+p_action[2]], c)
        fig = go.Figure(data=arrows)
        fig.show()



def _dep_graph_perturbations(mean_actions, perturbed_actions, forces=None, scores=None):
    fig = go.Figure()

    # fig.add_trace(go.Cone(
    #     x=mean_actions[:,0],
    #     y=mean_actions[:,1],
    #     z=mean_actions[:,2],
    #     u=mean_actions[:,3],
    #     v=mean_actions[:,4],
    #     w=mean_actions[:,5],
    #     colorscale='Greens',
    #     #color=np.linspace(0, 1, len(poses)),
    #     #sizemode="raw",
    #     sizeref=100,
    #     opacity=.1,
    # ))

    # if forces is not None:
    #     fig.add_trace(go.Cone(
    #         x=mean_actions[:,0],
    #         y=mean_actions[:,1],
    #         z=mean_actions[:,2],
    #         u=forces[:,0],
    #         v=forces[:,1],
    #         w=forces[:,2],
    #         colorscale='Reds',
    #         #color=np.linspace(0, 1, len(poses)),
    #         #sizemode="raw",
    #         sizeref=100,
    #         opacity=.1,
    #     ))

    if scores is not None:
        
        #print(perturbed_actions)
        #perturbed_actions = np.ones((16, 3))
        #perturbed_actions[:, 2] = np.linspace(-5,5, 16)
        #scores = [np.dot(np.array([0,0,1]), action) for action in perturbed_actions]
        scores -= np.min(scores)
        #scores *= 100
        #print(scores)
        scores = np.array(scores).reshape(1, -1)/np.max(scores) #* .95
        #print(scores.T)
        from plotly.express.colors import sample_colorscale

        for p_action, score in zip(perturbed_actions, scores.T):
            p_action= p_action.reshape(1, -1)
            p_norm = np.linalg.norm(p_action[:, :3])
            print(p_norm)



            c = sample_colorscale('Viridis', [score[0]])

            colorscale=[[0, f"rgb(1, 1, 1)"],
                    [p_norm, c[0],], 
                    [1, f"rgb(1, 1, 1)"],]
            
            
            fig.add_trace(go.Cone(
                x=p_action[:, 0],#np.concatenate([mean_actions[:,0] for _ in range(16)], 0) ,
                y=p_action[:, 1],#np.concatenate([mean_actions[:,1] for _ in range(16)], 0),
                z=p_action[:, 2],#np.concatenate([mean_actions[:,2] for _ in range(16)], 0),
                u=p_action[:, 0],#perturbed_actions[:,0] ,
                v=p_action[:, 1],#perturbed_actions[:,1],
                w=p_action[:, 2],#perturbed_actions[:,2],
                colorscale=colorscale,#,'Viridis',
                #color=np.linspace(0, 1, len(poses)),
                #sizemode="raw",
                sizeref=100,
                opacity=score[0]*.9,
            ))
    else:
        fig.add_trace(go.Cone(
            x=perturbed_actions[:,0],
            y=perturbed_actions[:,1],
            z=perturbed_actions[:,2],
            u=perturbed_actions[:,3],
            v=perturbed_actions[:,4],
            w=perturbed_actions[:,5],
            colorscale='Greys',
            #color=np.linspace(0, 1, len(poses)),
            #sizemode="raw",
            sizeref=100,
            opacity=.1,
        ))
    fig.show()


def graph_trajectory(poses, object_poses, im=None):
    fig = go.Figure()
    # im_x, im_y, im_layers = im.shape

    # eight_bit_img = Image.fromarray(im).convert('P', palette='WEB', dither=None)
    # dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')
    # idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
    # colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

    # x = np.linspace(0,im_x, im_x) * 1/im_x
    # y = np.linspace(0, im_y, im_y) * 1/im_y
    # z = np.zeros(im.shape[:2])

    #fig.write_html(titleString + "plot.html", auto_open=True) 
    #fig.show()
    print("SHOWED ONE")
    fig.add_trace(go.Scatter3d(
        x=poses[:,0],
        y=poses[:,1],
        z=poses[:,2],
        mode='markers',
        marker={'colorscale' : 'Greens',
                'color': np.linspace(0, 1, len(object_poses)),
                'opacity': .25}
    ))

    fig.add_trace(go.Cone(
        x=poses[:,0],
        y=poses[:,1],
        z=poses[:,2],
        u=poses[:,3],
        v=poses[:,4],
        w=poses[:,5],
        colorscale='Reds',
        #color=np.linspace(0, 1, len(poses)),
        #sizemode="raw",
        sizeref=100,
        #zopacity=.1,
    ))

    fig.add_trace(go.Scatter3d(
        x=object_poses[:,0],
        y=object_poses[:,1],
        z=object_poses[:,2],
        mode='markers',
        marker={'colorscale' : 'Blues',
                'color': np.linspace(0, 1, len(object_poses)),
                'opacity': .5}
    ))
    
    
    # fig.add_trace(go.Surface(x=x, y=y, z=z,
    #     surfacecolor=eight_bit_img, 
    #     cmin=0, 
    #     cmax=255,
    #     colorscale=colorscale,
    #     showscale=False,
    #     lighting_diffuse=1,
    #     lighting_ambient=1,
    #     lighting_fresnel=1,
    #     lighting_roughness=1,
    #     lighting_specular=0.5,

    # ))
    
    #fig.write_image('./test.png')
    fig.show()
    #plotly.offline.plot(fig)
    # print("SHOWED 2")

def _perturb_action(action):

    means = np.array([0,0,0,0,0,0])
    std = np.eye(6,6) * .01

    deltas = np.random.multivariate_normal(means, std, size=(16,))
    return deltas + action
    # #generate a bunch of perturbations to the aciton

    # max_perturbations = torch.tensor([.02, .02, .02, .1, .1, .1, 0])
    # min_perturbations = torch.tensor([.02, .02, .02, .1, .1, .1, 0])

    # sample_axes = torch.linspace(min_perturbations, max_perturbations, 5)
    # sample_space = torch.meshgrid(sample_axes)
    # return action + sample_space
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()

    PATH = "/home/rthompson/mimicgen_forcelearning/datasets/core/square_d0_force.hdf5"
    demo_path = args.folder
    hdf5_path = args.folder#os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")

    env_info = json.loads(f["data"].attrs["env_args"])
    env_info["env_name"] = env_info["env_name"]#.split("_")[0]

    del(env_info["env_version"])
    del(env_info["type"])

    env_info["env_kwargs"]["has_renderer"] = False#True
    env = robosuite.make(
        env_name=env_info["env_name"],
        **env_info["env_kwargs"],
        #single_object_mode=2,
        #nut_type="round",
        # has_renderer=True,
        # has_offscreen_renderer=True,
        # ignore_done=True,
        # use_camera_obs=True,
        # reward_shaping=True,
        # control_freq=20,
    )

    #env = ForceBinningWrapper(env)
    print(env._observables.keys())
    force_idx = list(env._observables.keys()).index('robot0_ee_force')
    torque_idx = list(env._observables.keys()).index('robot0_ee_torque')

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    num_images = 10
    was_success = False
    ep = random.choice(demos)

    obs = f["data/{}/obs".format(ep)]
    action = f["data/{}/actions".format(ep)][0]
    print(action)
    print(f["data/{}/actions".format(ep)][0])
    print()
    delta_actions = _perturb_action(action[:6])
    print(delta_actions)

    actions = np.array(f["data/{}/actions".format(ep)][2:3])
    
    forces =  f["data/{}/obs".format(ep)]['robot0_ee_force'][2:3]
    forces = np.zeros_like(forces)
    forces[:, 2]= np.ones_like(forces[:,2])
    perturbed_actions = []
    perturbed_action_scores = []
    for action, force in zip(actions, forces):
        perturbed_action = _perturb_action(action[:6])
        print(force)
        print(perturbed_action[:, :3])
        scores = np.matmul(force.reshape(1, -1), perturbed_action[:, :3].T)
        print(scores.T)
        perturbed_actions.append(perturbed_action)
        perturbed_action_scores.append(scores)
    


    perturbed_actions = np.concatenate(perturbed_actions, 0)
    perturbed_action_scores = np.concatenate(perturbed_action_scores, 1)
    # print(perturbed_actions.shape)
    # print(perturbed_action_scores.shape)


    perturbed_actions[:, 3:] = perturbed_actions[:, 3:]/np.linalg.norm(perturbed_actions[:, 3:], axis=1).reshape(-1, 1)
    #print(np.linalg.norm(perturbed_actions[:, 3:], axis=1))
    print(actions.shape)
    graph_perturbations(f["data/{}/actions".format(ep)][1, :3], actions, perturbed_actions, forces=forces, scores=perturbed_action_scores)

    #try to normalize the orientation??
    exit(0)

    pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
    ee_orns = np.concatenate([np.array(quat2euler(np.array(obs["robot0_eef_quat"][i]))).reshape(1,-1) for i in range(len(np.array(obs["robot0_eef_quat"])))], 0)
    
    poses = np.concatenate([np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_ee_force"])], axis=1)#ee_orns], axis=1)
    object_poses = np.array(obs['object'][:, :3])
    graph_trajectory(poses, object_poses)


