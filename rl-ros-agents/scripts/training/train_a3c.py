import rospy
from stable_baselines.common.vec_env import SubprocVecEnv
from rl_ros_agents.env_wappers.arena2dEnv import get_arena_envs, Arena2dEnvWrapper
from rl_ros_agents.utils.callbacks import SaveOnBestTrainingRewardCallback
from rl_ros_agents.utils import getTimeStr
from stable_baselines import A2C
from stable_baselines.common.policies import MlpLstmPolicy
import tensorflow as tf
import random
import numpy as np
import os
import sys
import argparse
# disable tensorflow deprecated information
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

LOGDIR = None
GAMMA = 0.95
LEARNING_RATE = 1e-3
N_STEPS = 4
MAX_GRAD_NORM = 0.1

TIME_STEPS = int(1e8)
REWARD_BOUND = 130
use_reward_bound = True


def main(log_dir = None,name_results_root_folder = "results",use_reward_bound = True,reward_bound = 130,gamma = 0.95,n_steps = 4, time_steps = int(1e5),max_grad_norm=0.1,learn_rate = 1e-3):
    args = parseArgs()

    # if log_dir doesnt created,use defaul one which contains the starting time of the training.
    if log_dir is None:
        if args.restart_training:
            #find the latest training folder
            latest_log_dir = os.path.join(name_results_root_folder,sorted(os.listdir(name_results_root_folder))[-1])
            logdir = latest_log_dir
        else:
            defaul_log_dir = os.path.join(name_results_root_folder, "A2C_" + getTimeStr())
            os.makedirs(defaul_log_dir, exist_ok=True)
            logdir = defaul_log_dir
    else:
        logdir = log_dir
    reward_bound = None if not use_reward_bound else reward_bound
    # get arena environments and custom callback
    envs = get_arena_envs(log_dir=logdir)
    call_back = SaveOnBestTrainingRewardCallback(500, logdir, 1, reward_bound)
    # set temporary model path, if training was interrupted by the keyboard, the current model parameters will be saved.
    path_temp_model = os.path.join(logdir,"A2C_TEMP")
    if not args.restart_training:
        model = A2C(MlpLstmPolicy, envs, verbose=1, n_steps=N_STEPS, max_grad_norm=MAX_GRAD_NORM,
                    learning_rate=LEARNING_RATE, gamma=GAMMA, tensorboard_log=logdir)
        reset_num_timesteps = True
    else:
        if os.path.exists(path_temp_model+".zip"):
            print("continue training the model...")
            model = A2C.load(path_temp_model,env=envs)
            reset_num_timesteps = False
        else:
            print("Can't load the model with the path: {}, please check again!".format(path_temp_model))
            envs.close()
            exit(-1)
    try:
        model.learn(time_steps, log_interval=200, callback=call_back,reset_num_timesteps=reset_num_timesteps)
        model.save(os.path.join(logdir, "A2C_final"))
    except KeyboardInterrupt:
        model.save(path_temp_model)
        print("KeyboardInterrupt: saved the current model to {}".format(path_temp_model))
    finally:
        envs.close()
        exit(0)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--restart_training",action="store_true",help="restart the latest unfinished training")
    return parser.parse_args()

if __name__ == "__main__":
    main(log_dir=LOGDIR,use_reward_bound=use_reward_bound,reward_bound=REWARD_BOUND,time_steps=TIME_STEPS,max_grad_norm=MAX_GRAD_NORM,gamma=GAMMA,learn_rate=LEARNING_RATE,n_steps=N_STEPS)
