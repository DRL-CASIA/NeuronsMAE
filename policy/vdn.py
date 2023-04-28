import torch
import os
from network.vdn_net import VDNNet
from network.qmix_net import myRNN
import time 

class VDN:
    def __init__(self, args, master):
        self.n_actions = args.n_actions
        self.n_agents_per_party = args.n_agents_per_party
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        self.master = master


        self.eval_rnn = myRNN(input_shape, args)
        self.target_rnn = myRNN(input_shape, args)
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()
        self.args = args
        if self.master and self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()
        stamp = int(time.time())
        if not self.args.dr_coef:
            self.model_dir = self.args.model_dir + '/' + self.args.action_space + '/' + args.alg + '/no_dr/' + '%f'%args.level  + '/' + (time.strftime("%Y_%m_%d_%H", time.localtime(stamp)))
        else:
            self.model_dir = self.args.model_dir + '/' + self.args.action_space + '/' + args.alg + '/dr/' + '%f'%args.level  + '/' + (time.strftime("%Y_%m_%d_%H", time.localtime(stamp)))
        # 如果存在模型则加载模型
        '''
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_vdn_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")
        '''

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        print('Init alg VDN')

    def learn(self, batch, train_step, target_update):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''

        transition_num = batch['o'].shape[0]
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']


        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数*max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch)
        if self.master and self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=2, index=u).squeeze(2)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=2)[0]

        q_total_eval = self.eval_vdn_net(q_evals)
        q_total_target = self.target_vdn_net(q_targets)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # if train_step > 0 and train_step % self.args.target_update_cycle == 0:
        if train_step == 0 and target_update:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    def get_para(self):
        return self.eval_rnn.state_dict()

    def load_para(self, rnn_para):
        self.eval_rnn.load_state_dict(rnn_para)
    
    
    def _get_inputs(self, batch):
        obs, obs_next = batch['o'], batch['o_next']
        transition_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)


        inputs = torch.cat([x.reshape(transition_num * 1 * self.args.n_agents_per_party, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(transition_num * 1 * self.args.n_agents_per_party, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next


    def get_q_values(self, batch):

        transition_num = batch['o'].shape[0]

        inputs, inputs_next = self._get_inputs(batch)  # 给obs加last_action、agent_id

        if self.master and self.args.cuda:
            inputs = inputs.cuda()
            inputs_next = inputs_next.cuda()

        q_eval = self.eval_rnn(inputs)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
        q_target = self.target_rnn(inputs_next)

        # 把q_eval维度重新变回(8, 5,n_actions)
        q_eval = q_eval.view(transition_num, 1 * self.n_agents_per_party, -1)
        q_target = q_target.view(transition_num, 1 * self.n_agents_per_party, -1)

        return q_eval, q_target



    def save_model(self, eps):
        # num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + str(eps) + '_vdn_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + str(eps) + '_rnn_net_params.pkl')

    def load_model(self, eps):
        if os.path.exists(self.model_dir + '/' + str(eps) + '_rnn_net_params.pkl'):
            path_rnn = self.model_dir + '/' + str(eps) + '_rnn_net_params.pkl'
            path_qmix = self.model_dir + '/' + str(eps) + '_vdn_net_params.pkl'
            map_location = 'cuda:0' if self.args.cuda else 'cpu'
            self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
            self.eval_vdn_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
            print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
        else:
            raise Exception("No model!")
