from typing import List, Dict, Optional, Optional
import numpy as np
import gymnasium
from gymnasium.spaces import Box
from diffusion_policy.env.kitchen.base import KitchenBase

class KitchenLowdimWrapper(gymnasium.Env):
    def __init__(self,
            env: KitchenBase,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            render_hw = (240,360)
        ):
        self.env = env
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.render_hw = render_hw

        # Wrap gym spaces as gymnasium spaces
        src_obs = env.observation_space
        src_act = env.action_space
        self.observation_space = Box(
            low=src_obs.low, high=src_obs.high,
            dtype=src_obs.dtype)
        self.action_space = Box(
            low=src_act.low, high=src_act.high,
            dtype=src_act.dtype)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.init_qpos is not None:
            _ = self.env.reset()
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env._get_obs()
        else:
            obs = self.env.reset()
        return obs, {}

    def render(self):
        h, w = self.render_hw
        return self.env.render(mode='rgb_array', width=w, height=h)

    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        return obs, reward, done, False, info
