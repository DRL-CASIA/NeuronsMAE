import numpy as np
import torch
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args, master):

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents_per_party
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.master = master
        print(args.alg)

        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args, self.master)
        elif args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args, self.master)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args, self.master)
        else:
            raise Exception("No such algorithm")

        self.args = args

    

    
    def choose_action(self, obs, agent_id, avail_actions, epsilon):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.master and self.args.cuda:
            inputs = inputs.cuda()

        # get q value
        q_value = self.policy.eval_rnn(inputs)
        q_value[avail_actions == 0.0] = - float("inf")

        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)

        return action


    def train(self, batch, train_step, target_update):  # coma needs epsilon for training
        self.policy.learn(batch, train_step, target_update)














