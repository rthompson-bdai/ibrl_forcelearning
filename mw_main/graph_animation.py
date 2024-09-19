
import plotly
import plotly.graph_objects as go

from os import listdir
from os.path import isfile, join
import pickle

import numpy as np
import io
from PIL import Image

from plotly.subplots import make_subplots
import imageio as iio
import moviepy.editor as mpy
import re
import torch

#sigh
def arrayify(list_of_tensors):
    replacement = []
    for t in list_of_tensors:
        if type(t) == torch.Tensor:
            replacement.append(t.squeeze().cpu().numpy())
        else:
            replacement.append(t)
    return np.array(replacement)


def get_images_from_folder(folder):
    files = [join(folder, f) for f in listdir(folder)]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    images = []
    for f in files: 
        im = iio.imread(f)
        images.append(im)
    return images

def get_force_from_pickle(data_file):
    data = pickle.load(open(data_file, 'rb'))
    return {'forces': np.array(data['forces'])[:, :3], 'torques': np.array(data['forces'])[:, 3:]}

def get_pose_from_pickle(data_file):
    data = pickle.load(open(data_file, 'rb'))
    poses = arrayify(data['prop'])[:, :3]
    return {'poses': poses}

def get_use_bcs_from_pickle(data_file):
    data = pickle.load(open(data_file, 'rb'))
    return {'use_bcs': arrayify(data['use_bcs'])}

def get_actions_from_pickle(data_file):
    data = pickle.load(open(data_file, 'rb'))
    return {'both_actions': arrayify(data['both_actions'])}


def plotly_fig2array(fig):
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def animate_actions(image_data, pose_data, action_data, use_bcs, force_data, title, filename):
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {'type': 'scene'}]])
    num_frames = len(image_data)
    end_frames = 0

    def make_arrow(x, y, z, color, name):
        datum = [
        {
            'x': x,
            'y': y,
            'z': z,
            'mode': "lines",
            'type': "scatter3d",
            'line': {
                'color': color,
                'width': 3},
            'name': f"line_{name}"
            
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
            'name': f"cone_{name}"
        }]

        traces = [go.Line(**datum[0]), go.Cone(**datum[1])]
        return traces

    BC_CHOSEN_COLOR = 'blue'
    BC_REJECTED_COLOR = 'gray'
    RL_CHOSEN_COLOR = 'red'
    RL_REJECTED_COLOR = 'gray'

    print(action_data['both_actions'])
    all_arrows = []
    for i, (pose, both_action, use_bc) in enumerate(zip(pose_data['poses'], action_data['both_actions'], use_bcs['use_bcs'])):
        #both_action = both_action.squeeze()
        print(pose)
        step_arrows = []
        if use_bc == 1.0:

            rl_arrow = make_arrow((pose[0], pose[0] + both_action[0, 0]), 
                                             (pose[1], pose[1] + both_action[0, 1]), 
                                             (pose[2], pose[2] + both_action[0, 2]), 
                                             RL_REJECTED_COLOR,
                                             f"rl_{i}")
            bc_arrow = make_arrow((pose[0], pose[0] + both_action[1, 0]), 
                                             (pose[1], pose[1] + both_action[1, 1]), 
                                             (pose[2], pose[2] + both_action[1, 2]), 
                                             BC_CHOSEN_COLOR,
                                             f"bc_{i}")
        else:
            rl_arrow = make_arrow((pose[0], pose[0] + both_action[0, 0]), 
                                             (pose[1], pose[1] + both_action[0, 1]), 
                                             (pose[2], pose[2] + both_action[0, 2]), 
                                             RL_CHOSEN_COLOR,
                                             f"rl_{i}")
            bc_arrow = make_arrow((pose[0], pose[0] + both_action[1, 0]), 
                                             (pose[1], pose[1] + both_action[1, 1]), 
                                             (pose[2], pose[2] + both_action[1, 2]), 
                                             BC_REJECTED_COLOR,
                                             f"bc_{i}")
        step_arrows += rl_arrow
        step_arrows += bc_arrow
        all_arrows.append(step_arrows)

    for frame in range(num_frames):
        arrows = all_arrows[frame]
        for trace in arrows:
            fig.add_trace(trace, row=1, col=2)

        fig.add_trace(go.Image(z=image_data[frame], name=f"Image {frame}"), row=1, col=1)

    def make_frame(t):
        t_index = int(t/2 * (num_frames + end_frames))
        arrow_frame_names = ["cone_bc_", "cone_rl_", "line_bc_", "line_rl_" ]
        
        for i in range(num_frames + end_frames):
            for name in arrow_frame_names:       
                fig.update_traces(visible=False, selector=dict(name=f"{name}{i}"))
            fig.update_traces(visible=False, selector=dict(name=f"Image {i}"))

        for i in range(t_index):
            for name in arrow_frame_names:       
                fig.update_traces(visible=True, selector=dict(name=f"{name}{i}"))
            
        fig.update_traces(visible=True, selector=dict(name=f"Image {t_index}"))
        fig.update_layout(title=f'{title}, t={t_index}', showlegend=False)
        fig.update_layout(scene2 = dict(xaxis=dict(range=[0,1.5],), yaxis=dict(range=[0,1.5],), zaxis=dict(range=[0,1.5],)))
        fig.layout.scene2.camera.eye=dict(x=2.4, y=2.4, z=2.4)

        return plotly_fig2array(fig)
    animation = mpy.VideoClip(make_frame, duration=2)
    animation.write_gif(f"{filename}.gif", fps=30)


def animate_force_only(force_datas, title, filename):
    fig = make_subplots(rows=6, cols=1,)# specs=[[{"rowspan": 6}, {}],[None, {}],[None, {}],[None, {}],[None, {}],[None, {}]])

    num_frames = max([len(force_datas[i]['forces']) for i in range(len(force_datas))])
    end_frames = 2

    #print(force_datas)

    x_labels = ["X", "Y", "Z", "R", "P", "Y"]
    for i in range(3):
        fig.update_xaxes(row=i+1, col=1, range=[0, num_frames], tickfont = dict(size=7)) 
        fig.update_yaxes(title_text=x_labels[i], title_standoff=2,tickfont = dict(size=7), row=i+1, col=1, range=[-200,200], tickmode = 'linear', tick0 = -200, dtick = 200) 
    for i in range(3):
        fig.update_xaxes(row=i+1+3, col=1, range=[0, num_frames], tickfont = dict(size=7)) 
        fig.update_yaxes(title_text=x_labels[i+3], title_standoff=2,tickfont = dict(size=7), row=i+1+3, col=1, range=[-20,20], tickmode = 'linear', tick0 = -10, dtick = 10) 

    for i, force_data in enumerate(force_datas):
        for frame in range(len(force_data['forces'])):

            forces = force_data['forces'][:frame]
            torques = force_data['torques'][:frame]

            #fig.add_trace(go.Image(z=image_data[frame], name=f"Image {frame}")) #image
            
            fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name=f'X Force {frame}_{i}'), row=1, col=1)
            fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name=f'Y Force {frame}_{i}'), row=2, col=1)
            fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name=f'Z Force {frame}_{i}'), row=3, col=1)
            fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name=f'R Torque {frame}_{i}'), row=4, col=1)
            fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name=f'P Torque {frame}_{i}'), row=5, col=1)
            fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name=f'Y Torque {frame}_{i}'), row=6, col=1)

    # for i, force_data in enumerate(force_datas):
    #     for force_data in force_datas:
    #         for frame in range(end_frames):
    #             #fig.add_trace(go.Image(z=image_data[-1], name=f"Image {num_frames + frame}")) #image
                
    #             fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name=f'X Force {num_frames + frame}_{i}'), row=1, col=1)
    #             fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name=f'Y Force {num_frames + frame}_{i}'), row=2, col=1)
    #             fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name=f'Z Force {num_frames + frame}_{i}'), row=3, col=1)
    #             fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name=f'R Torque {num_frames + frame}_{i}'), row=4, col=1)
    #             fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name=f'P Torque {num_frames + frame}_{i}'), row=5, col=1)
    #             fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name=f'Y Torque {num_frames + frame}_{i}'), row=6, col=1)


    max_frames = [len(force_datas[i]['forces']) for i in range(len(force_datas))]
    def make_frame(t):
        t_index = int(t/2 * (num_frames))
        force_frame_names = ["X Force ", "Y Force ", "Z Force ", "R Torque ", "P Torque ", "Y Torque ", ]
        
        for i in range(num_frames):
            for j in range(len(force_datas)):
                for name in force_frame_names:       
                    fig.update_traces(visible=False, selector=dict(name=f"{name}{i}_{j}"))
                #fig.update_traces(visible=False, selector=dict(name=f"Image {i}_{j}"))

        for name in force_frame_names:
            for j in range(len(force_datas)):
                fig.update_traces(visible=True, selector=dict(name=f"{name}{t_index}_{j}"))
                if t_index >= max_frames[j]:
                    fig.update_traces(visible=True, selector=dict(name=f"{name}{max_frames[j]-1}_{j}"))

                #fig.update_traces(visible=True, selector=dict(name=f"Image {t_index}_{j}"))
        fig.update_layout(title=f'{title}, t={t_index}', showlegend=False)
        return plotly_fig2array(fig)

    animation = mpy.VideoClip(make_frame, duration=2)
    animation.write_gif(f"{filename}.gif", fps=10)#30)


def animate_force(image_data, force_data, title, filename):
    fig = make_subplots(rows=6, cols=2, specs=[[{"rowspan": 6}, {}],[None, {}],[None, {}],[None, {}],[None, {}],[None, {}]])
    num_frames = len(image_data)
    end_frames = 2

    x_labels = ["X", "Y", "Z", "R", "P", "Y"]
    for i in range(3):
        fig.update_xaxes(row=i+1, col=2, range=[0, num_frames], tickfont = dict(size=7)) 
        fig.update_yaxes(title_text=x_labels[i], title_standoff=2,tickfont = dict(size=7), row=i+1, col=2, range=[-100,100], tickmode = 'linear', tick0 = -200, dtick = 200) 
    for i in range(3):
        fig.update_xaxes(row=i+1+3, col=2, range=[0, num_frames], tickfont = dict(size=7)) 
        fig.update_yaxes(title_text=x_labels[i+3], title_standoff=2,tickfont = dict(size=7), row=i+1+3, col=2, range=[-20,20], tickmode = 'linear', tick0 = -10, dtick = 10) 

    for frame in range(num_frames):

        forces = force_data['forces'][:frame]
        torques = force_data['torques'][:frame]

        fig.add_trace(go.Image(z=image_data[frame], name=f"Image {frame}")) #image
        
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name=f'X Force {frame}'), row=1, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name=f'Y Force {frame}'), row=2, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name=f'Z Force {frame}'), row=3, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name=f'R Torque {frame}'), row=4, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name=f'P Torque {frame}'), row=5, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name=f'Y Torque {frame}'), row=6, col=2)

    for frame in range(end_frames):
        fig.add_trace(go.Image(z=image_data[-1], name=f"Image {num_frames + frame}")) #image
        
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name=f'X Force {num_frames + frame}'), row=1, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name=f'Y Force {num_frames + frame}'), row=2, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name=f'Z Force {num_frames + frame}'), row=3, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name=f'R Torque {num_frames + frame}'), row=4, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name=f'P Torque {num_frames + frame}'), row=5, col=2)
        fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name=f'Y Torque {num_frames + frame}'), row=6, col=2)


    def make_frame(t):
        
        t_index = int(t/2 * (num_frames + end_frames))
        force_frame_names = ["X Force ", "Y Force ", "Z Force ", "R Torque ", "P Torque ", "Y Torque ", ]
        
        for i in range(num_frames + end_frames):
            for name in force_frame_names:       
                fig.update_traces(visible=False, selector=dict(name=f"{name}{i}"))
            fig.update_traces(visible=False, selector=dict(name=f"Image {i}"))

        for name in force_frame_names:
            fig.update_traces(visible=True, selector=dict(name=f"{name}{t_index}"))
            fig.update_traces(visible=True, selector=dict(name=f"Image {t_index}"))
        fig.update_layout(title=f'{title}, t={t_index}', showlegend=False)
        return plotly_fig2array(fig)

    animation = mpy.VideoClip(make_frame, duration=2)
    animation.write_gif(f"{filename}.gif", fps=10)#30)

        # fig.save thing

if __name__ == "__main__":


    # #image_data = get_images_from_folder(f"bc_viz_no_force/metaworld/faucet-open_floor_texture/episode{0}")
    # force_datas = [get_force_from_pickle(f"rl_viz_force/metaworld/faucet-open_floor_texture/episode{i}.mp4.reward.pkl") for i in range(5)]


    # animate_force_only(force_datas, "Faucet-Open", f"force_profile_animations/faucet_force_stack")
    # #animate_force(image_data, force_data, "Faucet-Open", f"force_profile_animations/faucet_open_successs{0}")


    image_data = get_images_from_folder("bc_viz_no_force/metaworld/faucet-open_table_texture/episode6")
    force_data = get_force_from_pickle("bc_viz_no_force/metaworld/faucet-open_arm_pos/episode6.mp4.reward.pkl")

    animate_force(image_data, force_data, "Faucet-Open", "force_profile_animations/faucet-open_6")



    # image_data = get_images_from_folder("rl_viz_force/metaworld/button-press_arm_pos/episode0")
    # force_data = get_force_from_pickle("rl_viz_force/metaworld/button-press_arm_pos/episode0.mp4.reward.pkl")
    # pose_data = get_pose_from_pickle("rl_viz_force/metaworld/button-press_arm_pos/episode0.mp4.reward.pkl")
    # use_bcs_data = get_use_bcs_from_pickle("rl_viz_force/metaworld/button-press_arm_pos/episode0.mp4.reward.pkl")
    # action_data = get_actions_from_pickle("rl_viz_force/metaworld/button-press_arm_pos/episode0.mp4.reward.pkl")

    # animate_actions(image_data, pose_data, action_data, use_bcs_data, force_data, "Door-Open", "action_animations/door-open")







