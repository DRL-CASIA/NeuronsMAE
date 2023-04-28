import torch.nn as nn
import torch
import torch.nn.functional as F


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        '''
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, 1 * args.n_agents_per_party * 1))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            # self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
            #                               nn.ReLU(),
            #                               nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        # else:
        #     self.hyper_w1 = nn.Linear(args.state_shape, 1 * args.n_agents_per_party * args.qmix_hidden_dim)
        #     # 经过hyper_w2得到(经验条数, 1)的矩阵
        #     self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, 1)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        # self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
        #                              nn.ReLU(),
        #                              nn.Linear(args.qmix_hidden_dim, 1)
        #                              )
        '''
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents_per_party * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents_per_party * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1))




    def forward(self, q_values, states): 
        # q_values (tran_num, n_agents)
        
        episode_num = q_values.size(0)

        '''
        q_values = q_values.view(-1, 1, self.args.n_agents_per_party)  # (tran_num, 1, n_agents) 
        

        states = states.reshape(-1, self.args.state_shape)  # (tran_num, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # (tran_num, 1 * nagents * 1)
        # print(w1.shape)
        # b1 = self.hyper_b1(states)  ## (tran_num, 1)

        w1 = w1.view(-1, 1 * self.args.n_agents_per_party, 1)  # (tran_num, 1 * nagents, 1)
        # print(w1.shape)
        # b1 = b1.view(-1, 1, 1)  # (tran_num, 1, 1)

        # hidden = F.elu(torch.bmm(q_all, w1) + b1)  # (tran_num, 2 * nagents, 1)
        hidden = torch.bmm(q_values, w1)  # (tran_num, 1, 1) 

        q_total_ = hidden.squeeze(2)  # (tran_num, 1)
        '''
        # q_total_ = torch.split(q_values, 1, 1)[0] + torch.split(q_values, 1, 1)[1]

        # episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents_per_party)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents_per_party, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        # q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        q_total_ = q_total.squeeze(2) 

        return q_total_

class myRNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(myRNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        h = F.relu(self.fc3(x))
        q = self.fc2(h)
        return q