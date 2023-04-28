from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from RmCooAbsEnv import RmCooAbsEnv
from RmCooDisEnv import RmCooDisEnv

if __name__ == '__main__':
    
    args = get_common_args()
    args = get_mixer_args(args)


    if(args.action_space == "discret"):
        env = RmCooDisEnv()
    else:
        env = RmCooAbsEnv()
    
        
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents_per_party = env_info["n_agents_per_party"] 
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    env.close()
    runner = Runner(args) 
    runner.run()


