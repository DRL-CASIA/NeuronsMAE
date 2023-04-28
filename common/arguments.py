import argparse

"""
Here are the param for the training

"""


def get_common_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_num', type=int, default=20, help='number of the workers')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default='vdn', help='the algorithm to train the agent')
    parser.add_argument('--n_eps', type=int, default=1501, help='total eps')
    # parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--print_cycle', type=int, default=20, help='how often to evaluate the model, every 100 ep')
    parser.add_argument('--evaluate_cycle', type=int, default=50, help='how often to evaluate the model, every 100 ep')
    parser.add_argument('--evaluate_epoch', type=int, default=10, help='number of the epoch to evaluate the agent, test 100 ep')
    parser.add_argument('--episode_limit_test', type=int, default=50, help='number of the step to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory of the policy')
    parser.add_argument('--load_model', default=False, action="store_true", help='whether to load the pretrained model')
    parser.add_argument('--cuda', default=False, action="store_true", help='whether to use the GPU')
    parser.add_argument('--test', default=False, action="store_true", help='whether to test')
    parser.add_argument('--no_graphics', default=False, action="store_true", help='whether no_graphics')
    parser.add_argument('--time_scale', type=int, default=100, help='time_scale')
    parser.add_argument('--action_space', type=str, default='discret', help='action_space')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--level', type=float, default=0.0, help='level')
    parser.add_argument('--dr_coef', default=False, action="store_true", help='whether dr_VK1')
    parser.add_argument('--coef_tr', type=float, default=0.5, help='VK1')
    parser.add_argument('--coef_ts', type=float, default=1.5, help='VK1')

    args = parser.parse_args()
    return args




# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 128
    args.hyper_hidden_dim = 32              # betwwen hyper layer and inputs
    args.qmix_hidden_dim = 32               # betwwen two hyper layers 

    args.two_hyper_layers = False

    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 0.2
    args.min_epsilon = 0.05
    anneal_steps = int(6e2)
    # anneal_steps = int(1000)

    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'episode'

    # the number of the train steps in one epoch
    args.train_steps = 30

    # experience replay
    args.batch_size = 2000
    # args.buffer_size = int(1e5)
    args.buffer_size = int(2e5)

    # how often to save the model
    args.save_cycle = 50

    # how often to update the target_net
    args.target_update_cycle = 5

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args




