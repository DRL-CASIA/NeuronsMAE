from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
import numpy as np
import gym
import random
import os
import time



class RmCooAbsEnv():

    def __init__(self, seed=1234, port=10000, no_graphics=True, time_scale = 100):
        self.seed = seed + port
        self.port = port
        self.env = Unity3DEnv('coo_a_vs_a/coo_a_vs_a.x86_64',
                              seed = self.seed, port = self.port, 
                              no_graphics=no_graphics, time_scale=time_scale)
        print(time_scale)
        self.obs_size = 37
        self.act_size = 8
        # self.group_name = list(self.env.unity_env.behavior_specs.keys())
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,))
        self.action_space = gym.spaces.Discrete(self.act_size)
        self.agent_name = []
        self.obs = None
        self.state = None
        obs_dict = self.env.reset(None)
        for k,v in obs_dict.items():
            self.agent_name.append(k)
        pass

    def reset(self, dy_para_dict):  
        obs_dict = self.env.reset(dy_para_dict)
        obs = []
        for k,v in obs_dict.items():
            obs.append(v)
        self.obs = obs
        return obs

    def step(self, action):

        act_dict = {}

        for i, name in enumerate(self.agent_name):
            act_dict[name] = np.array([action[i]//4, action[i]%4])
        
        obs_dict, rewards_dict, dones, infos = self.env.step(act_dict)
  
        obs = []
        for k,v in obs_dict.items():
            obs.append(v)

        rewards = []
        for k,v in rewards_dict.items():
            rewards.append(v)
        
        self.obs = obs

        return obs, rewards, dones, infos
    
    def get_obs(self):
        return self.obs
    
    def get_state(self):
        self.state = []
        self.state.append(self.obs[0])
        return self.state

    def get_env_info(self):
        env_info = {"n_actions": 8, "n_agents_per_party": 2, "state_shape": 37, "obs_shape": 37, "episode_limit": 50}
        return env_info
    
    def close(self):
        self.env.close()
        pass


# env = RmCooAbsEnv()
# dy_para_dict = {"VK1": 0.0375, "level": 0}
# obs = env.reset(dy_para_dict)
# print(obs)
   
