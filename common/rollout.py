import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
from agent.agent import Agents
from RmCooAbsEnv import RmCooAbsEnv
from RmCooDisEnv import RmCooDisEnv

class RolloutWorker:
    def __init__(self, id, args):
        self.args = args
        self.id = id

        if(args.action_space == "discret"):
            self.env = RmCooDisEnv(seed = self.args.seed, port=10000+self.id, no_graphics=self.args.no_graphics, time_scale = self.args.time_scale)
        else:
            self.env = RmCooAbsEnv(seed = self.args.seed, port=10000+self.id, no_graphics=self.args.no_graphics, time_scale = self.args.time_scale)
       
        self.agents = Agents(self.args, False)


        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents_per_party
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        # self.episode_limit_test = args.episode_limit_test
        print('Init RolloutWorker')



    def generate_episode(self, model):

        self.agents.policy.load_para(model)

        o,  u,  r,  s,  avail_u,  u_onehot   = [], [], [], [], [], []
         
        terminate, padded = [], []

        if(self.args.dr_coef):
            coef_list = np.array([0.5, 0.7, 1, 1.4, 1.7, 2]) 
            dy_para_dict = {"coef": np.random.choice(coef_list), "level": self.args.level}
        else:
            dy_para_dict = {"coef": self.args.coef_tr, "level": self.args.level}
        
        obs = self.env.reset(dy_para_dict)

        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  


        # epsilon
        epsilon = self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        self.epsilon = epsilon
        max_episode_limit = self.episode_limit + 1

    
        while not terminated:

            obs = self.env.get_obs()

            state = self.env.get_state()

            actions, avail_actions, actions_onehot = [], [], []

            num_agent = 1 * self.args.n_agents_per_party

            for agent_id in range(num_agent):

                avail_action = [1. for _ in range(self.args.n_actions)]
                action  = self.agents.choose_action(obs[agent_id], agent_id, avail_action, epsilon)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)


            obs2, reward, dones, info = self.env.step(actions)
            terminated = dones
            
           
            o.append(obs)
            s.append(state[0])
            u.append(np.reshape(actions, [1 * self.args.n_agents_per_party, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward[0]])
            terminate.append([1. if dones else 0.])
            padded.append([0.])
            episode_reward += reward[0]
            step += 1



        # last obs
        episode = None

        obs = self.env.get_obs()
        state = self.env.get_state()

        o.append(obs)

        s.append(state[0])

        o_next  = o[1:]

        s_next  = s[1:]

        o  = o[:-1]

        s  = s[:-1]


        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions  = []

        for agent_id in range(1 * self.args.n_agents_per_party):
            avail_action = [1 for _ in range(self.args.n_actions)]
            avail_actions.append(avail_action)

        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        episode = dict( o=o.copy(),
                        s=s.copy(),
                        u=u.copy(),
                        r=r.copy(),
                        avail_u=avail_u.copy(),
                        o_next=o_next.copy(),
                        s_next=s_next.copy(),
                        avail_u_next=avail_u_next.copy(),
                        u_onehot=u_onehot.copy(),
                        padded=padded.copy(),
                        terminated=terminate.copy() )
 
        # add episode dim
        for key in episode.keys():
            episode[key] = np.squeeze(np.array([episode[key]]), 0)
   
        win_tag = 1 if reward[0]>13 else 0 
     
        return episode, episode_reward, win_tag, step




    def generate_episode_test(self, model):

        self.agents.policy.load_para(model)

        o,  u,  r,  s,  avail_u,  u_onehot   = [], [], [], [], [], []
         
        terminate, padded = [], []

        dy_para_dict = {"coef": self.args.coef_ts, "level": self.args.level}
        obs = self.env.reset(dy_para_dict)
        terminated = False

        win_tag = False

        step = 0
        episode_reward = 0  


        # epsilon
        epsilon = 0 

        max_episode_limit = self.episode_limit


        while not terminated:
            obs = self.env.get_obs()

            state = self.env.get_state()

            actions, avail_actions, actions_onehot = [], [], []

            num_agent = 1 * self.args.n_agents_per_party

            for agent_id in range(num_agent):

                avail_action = [1. for _ in range(self.args.n_actions)]
                action  = self.agents.choose_action(obs[agent_id], agent_id, avail_action, epsilon)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)

            obs2, reward, terminated, info = self.env.step(actions)
           
            episode_reward += reward[0]
            step += 1



        win_tag = 1 if reward[0]>13 else 0 
        
        return win_tag, episode_reward, step

   
   