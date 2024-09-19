import pickle
import torch
import numpy as np

from env.metaworld_wrapper import PixelMetaWorld
from common_utils import ibrl_utils as utils
from common_utils import Recorder

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os


def graph_force(force_data):
    fig = make_subplots(rows=3, cols=1, subplot_titles=("EE Force", "Demo", "EE Torque"), )
    forces=np.array(np.array(force_data)[:, :3])#v_func_force(np.array(forces) - initial_force)
    torques=np.array(np.array(force_data)[:, 3:])#v_func_torque(np.array(torques))
    
    fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,0], name='rX force'), row=1, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,1], name='rY force'), row=1, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(forces)), y= forces[:,2], name='rZ force'), row=1, col=1)

    # #print(dir(go))
    # #figm = px.imshow(images, binary_string=True, facet_col=0, facet_col_wrap=10)
    # #fig.add_trace(figm.data[0], 2, 1)
    # #print(len(images))
    # for i, image in enumerate(images):
    #     fig.add_trace( go.Image(z=np.flipud(image), x0=i*len(image),), row=2, col=1)

    fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,0], name='lX force'), row=3, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,1], name='lY force'), row=3, col=1)
    fig.add_trace(go.Scatter(x = np.arange(len(torques)), y= torques[:,2], name='lZ force'), row=3, col=1)
    return fig

def run_eval(
    env: PixelMetaWorld,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    eval_mode=True,
    save_video=True,
    save_plots=True,
    save_images=False,
    only_failures=False, 
):
    recorder = None if record_dir is None else Recorder(record_dir)

    scores = []
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            forces = []
            prop = []
            use_bcs = []
            actions = []
            qas = []
            both_actions = []

            np.random.seed(seed + episode_idx)
            #agent.return_stats = True
            obs, image_obs = env.reset()

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                obs['prop'] = obs['prop'].float()
                forces.append(obs['prop'][-6:].cpu().numpy())
                prop.append(obs["state"][:4].cpu().numpy())
                agent.return_stats = True
                action = agent.act(obs, eval_mode=eval_mode)
                if type(action) == tuple:
                    action, use_bc, qa, both_action = action
                else:
                    use_bc, qa, both_action = None, None, None
                agent.return_stats = False
                action = action.cpu().numpy()
                actions.append(action)
                use_bcs.append(use_bc)#.cpu().numpy())
                qas.append(qa)#).cpu().numpy())
                both_actions.append(both_action)#.cpu().numpy())


                obs, reward, terminal, _, image_obs = env.step(action)
                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))
            factors = env.unwrapped.factors
            if recorder is not None:
                save_path = recorder.save(f"episode{episode_idx}", save_video=save_video, save_images=save_images)
                if save_plots:
                    fig = graph_force(forces)
                    fig.write_image(os.path.join(record_dir, f"episode{episode_idx}_force_profile.png"))
                    
                reward_path = f"{save_path}.reward.pkl"
                print(f"saving reward to {reward_path}")
                pickle.dump({'rewards': rewards, \
                            'factors': factors, 
                            'actions': actions,
                            "use_bcs": use_bcs,
                            "qas": qas, 
                            "prop": prop,
                            "both_actions": both_actions, 
                            "forces": forces, }, open(reward_path, "wb"))

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")

    return scores
