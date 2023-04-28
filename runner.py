import numpy as np
import os
import ray 
import time 
import matplotlib.pyplot as plt
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

ray.init()

@ray.remote(num_cpus=3)
class RayRolloutWorker():
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.rayrolloutWorker = RolloutWorker(self.id, self.args)
    def rollout(self, model):
        return self.rayrolloutWorker.generate_episode(model)
    def rollout_test(self, model):
        return self.rayrolloutWorker.generate_episode_test(model)
    def close(self):
        pass



class Runner:
    def __init__(self, args):

        self.args = args

        self.agents = Agents(args, master = True)

        self.buffer = ReplayBuffer(self.args)

        self.workers = None
        
        stamp = int(time.time())
        if not self.args.dr_coef:
            self.save_path = self.args.result_dir + '/' + args.alg + '/no_dr/' + '%f'%args.level + '/' + (time.strftime("%Y_%m_%d_%H", time.localtime(stamp)))
        else:
            self.save_path = self.args.result_dir + '/' + args.alg + '/dr/' + '%f'%args.level + '/' + (time.strftime("%Y_%m_%d_%H", time.localtime(stamp)))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        
        if not self.args.dr_coef:
            self.log_path = self.args.log_dir + '/' + self.args.action_space + '/' + args.alg + '/no_dr/' + '%f'%args.level  + '/' + (time.strftime("%Y_%m_%d_%H", time.localtime(stamp)))
        else:
            self.log_path = self.args.log_dir + '/' + self.args.action_space + '/' + args.alg + '/dr/' + '%f'%args.level  + '/' + (time.strftime("%Y_%m_%d_%H", time.localtime(stamp)))
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = SummaryWriter(self.log_path)

        np.random.seed(self.args.seed)
        
        

    def run(self):
        self.workers = [RayRolloutWorker.remote(i, self.args) for i in range(1, self.args.env_num + 1)]
        
        time_steps = 0
        episode_rewards_train = 0
        is_wins = 0

        for eps in range(int(self.args.n_eps)):
            
            rnn_para = self.agents.policy.get_para()

            model_id = ray.put(rnn_para)
        
            buffers_ids = [
                worker.rollout.remote(model_id) for worker in self.workers
            ]

            start_time = time.time()
            episodes = []
        
            for batch in range(self.args.env_num):
                [buffers_id], buffers_ids = ray.wait(buffers_ids)
                episode, episode_reward_train, is_win, steps = ray.get(buffers_id)
            
                episodes.append(episode)
                time_steps += steps
                is_wins += is_win
                episode_rewards_train += episode_reward_train

            if (eps % self.args.print_cycle == 0) : 
                if eps != 0 :
                    print("ep : ", eps, " step : ", time_steps, 
                        " win_rate : ", is_wins / self.args.env_num / self.args.print_cycle, 
                        "  reward : ",episode_rewards_train / self.args.env_num/ self.args.print_cycle)
                    
                    self.writer.add_scalar('train_win_rate', is_wins / self.args.env_num / self.args.print_cycle, global_step = eps * self.args.env_num)
                    self.writer.add_scalar('train_rewards', episode_rewards_train / self.args.env_num / self.args.print_cycle, global_step = eps * self.args.env_num)
                else:
                    print("ep : ", eps, " step : ", time_steps, 
                        " win_rate : ", is_wins / self.args.env_num, 
                        "  reward : ",episode_rewards_train / self.args.env_num)
                    
                    self.writer.add_scalar('train_win_rate', is_wins / self.args.env_num, global_step = eps * self.args.env_num)
                    self.writer.add_scalar('train_rewards', episode_rewards_train / self.args.env_num, global_step = eps * self.args.env_num)
                
                episode_rewards_train = 0
                is_wins = 0

            episode_batch = episodes[0]
            episodes.pop(0)

            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # train qmix here
            self.buffer.store_episode(episode_batch)

            if time_steps > 0:
                need_train_step = 2 * int (self.buffer.current_size // self.args.batch_size)
                # if need_train_step < 20:
                if need_train_step < 20:
                    need_train_step = 20

                if eps % self.args.target_update_cycle == 0:
                    target_update = True
                else:
                    target_update = False

                for train_step in range(need_train_step):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_step, target_update)


            if eps % self.args.save_cycle == 0:
                self.agents.policy.save_model(eps)

            
            if eps % self.args.evaluate_cycle == 0:

                rnn_para = self.agents.policy.get_para()

                model_id = ray.put(rnn_para)
            
                win_rates = 0
                episode_rewards = 0
                ep_steps = 0

                for i in range(self.args.evaluate_epoch):

                    buffers_ids = [worker.rollout_test.remote(model_id) for worker in self.workers]

                    for batch in range(self.args.env_num):
                    
                        [buffers_id], buffers_ids = ray.wait(buffers_ids)
                        
                        win_rate, episode_reward, ep_step = ray.get(buffers_id)
                        
                        win_rates += win_rate
                        episode_rewards += episode_reward
                        ep_steps += ep_step
                

                print(  "testing ...... ep : ", eps, 
                        " win_rate : ", win_rates / self.args.env_num / self.args.evaluate_epoch, 
                        "  reward : ",episode_rewards / self.args.env_num / self.args.evaluate_epoch)
                
                
                self.writer.add_scalar('test_win_rate', win_rates / self.args.env_num / self.args.evaluate_epoch, global_step = eps * self.args.env_num)
                self.writer.add_scalar('test_rewards', episode_rewards / self.args.env_num / self.args.evaluate_epoch, global_step = eps * self.args.env_num)


            



    # def test(self, port, eps):
    #     self.agents.policy.load_model(eps)
    #     print("self.args.no_graphics  :       *********************************              ", self.args.no_graphics)
    #     rolloutWorker = RolloutWorker(port, self.args, no_graphics=self.args.no_graphics, time_scale = self.args.time_scale)
    #     rnn_para = self.agents.policy.get_para()
    #     win_rates = 0
    #     episode_rewards = 0
    #     ep_steps = 0
    #     for i in range(20):
    #         print(i)
    #         win_rate, episode_reward, ep_step = rolloutWorker.generate_episode_test_easy(rnn_para)
    #         win_rates += win_rate
    #         episode_rewards += episode_reward
    #         ep_steps += ep_step
    #     print(  "testing ...... ep : ", eps, 
    #             " win_rate : ", win_rates / 20, 
    #             "  reward : ",episode_rewards / 20)





