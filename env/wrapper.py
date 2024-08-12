"""
A collection of useful environment wrappers.
"""
from copy import deepcopy
import textwrap
import numpy as np
import torch
from collections import deque

import robomimic.envs.env_base as EB


class EnvWrapper(object):
    """
    Base class for all environment wrappers in robomimic.
    """
    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        #assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, EnvWrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(
                            self.__class__.__name__
                        )
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''

    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr


class ForceBinningWrapper(EnvWrapper):
    def __init__(self, env):
        super(ForceBinningWrapper, self).__init__(env=env)
        def force_binning(force):
            if force > 5:
                return 1
            if force < -5:
                return -1
            return 0

        def torque_binning(torque):
            if torque > 0.5:
                return 1
            if torque < -0.5:
                return -1
            return 0

        self.force_bin = np.vectorize(force_binning)
        self.torque_bin = np.vectorize(torque_binning)

    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        num_action = actions.size(0)

        rl_obs = {}
        # record the action in original format from model
        if self.cond_action > 0:
            for i in range(actions.size(0)):
                self.past_actions.append(actions[i])
            past_action = torch.stack(list(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        high_res_images = {}
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _ = self.env.env.step(actions[i])
            obs['robot0_ee_force'] = self.force_bin(obs['robot0_ee_force'] - self.meanshift_force)
            obs['robot0_ee_torque'] = self.torque_bin(obs['robot0_ee_torque'])
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            curr_rl_obs, curr_high_res_images = self._extract_images(obs)

            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                high_res_images.update(curr_high_res_images)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, high_res_images

    # def step(self, action):
    #     observation, r, done, info = self.env.step(action)
    #     observation['robot0_ee_force'] = self.force_bin(observation['robot0_ee_force'] - self.meanshift_force)
    #     observation['robot0_ee_torque'] = self.torque_bin(observation['robot0_ee_torque'])

        
    #     return observation, r, done, info

    def reset(self):
        obs = self.env.reset()
        obs, _, _, _ = self.env.env.step([0,0,0,0,0,0,1])
        curr_rl_obs, curr_high_res_images = self.env._extract_images(obs)
        self.meanshift_force = obs['robot0_ee_force']
        return curr_rl_obs, None

    def reset_to(self, state):
        obs = self.env.reset_to(state)
        obs, _, _, _= self.env.env.step([0,0,0,0,0,0,1])
        curr_rl_obs, curr_high_res_images = self.env._extract_images(obs)
        self.meanshift_force = obs['robot0_ee_force']
        return curr_rl_obs, None

