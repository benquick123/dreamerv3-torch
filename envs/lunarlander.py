import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLanderContinuous
import numpy as np


class LunarLander:
    def __init__(self):
        self._env = gym.make("LunarLander-v2", continuous=True)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = "obs"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return gym.spaces.Dict(
            {
                **spaces,
                "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = False
        return space

    def step(self, action):
        obs, reward, terminal, truncated, info = self._env.step(action)
        # print(obs, reward, done, info)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
            
        if reward == 100:
            reward = 1.0
            done = False
        elif reward == -100:
            reward = -1.0
        else:
            reward = 0.0
            
        self._step += 1
        
        obs["reward"] = reward
        obs["is_first"] = False
        obs["is_last"] = terminal
        obs["is_terminal"] = self._step == 500
        return obs, reward, terminal, info

    def reset(self):
        obs, info = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self._step = 0
        return obs
