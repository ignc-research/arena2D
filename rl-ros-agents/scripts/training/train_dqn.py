import rospy
from stable_baselines.common.vec_env import SubprocVecEnv
from rl_ros_agents.env_wappers.arena2dEnv import get_arena_envs, Arena2dEnvWrapper
from rl_ros_agents.utils.callbacks import SaveOnBestTrainingRewardCallback
from rl_ros_agents.utils import getTimeStr
from stable_baselines import DQN
from stable_baselines.common.policies import MlpLstmPolicy, FeedForwardPolicy
from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines.bench import Monitor
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
LEARNING_RATE = 0.00025
BUFFER_SIZE = 1000000
SYNC_TARGET_STEPS = 2000
N_STEPS = 4
MAX_GRAD_NORM = 0.1
BATCH_SIZE = 64

TIME_STEPS = int(1e8)
REWARD_BOUND = 130


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[64, 64, 64],
                                           act_fun=tf.nn.relu,
                                           feature_extraction="mlp")


def main(log_dir=None, name_results_root_folder="results"):
    args = parseArgs()
    time_steps = TIME_STEPS
    # if log_dir doesnt created,use defaul one which contains the starting time of the training.
    if log_dir is None:
        if args.restart_training:
            # find the latest training folder
            latest_log_dir = os.path.join(name_results_root_folder, sorted(os.listdir(name_results_root_folder))[-1])
            logdir = latest_log_dir
        else:
            defaul_log_dir = os.path.join(name_results_root_folder, "DQN_" + getTimeStr())
            os.makedirs(defaul_log_dir, exist_ok=True)
            logdir = defaul_log_dir
    else:
        logdir = log_dir
    reward_bound = REWARD_BOUND
    # get arena environments and custom callback
    env = Monitor(Arena2dEnvWrapper(0, True), os.path.join(logdir, "arena_env0"))
    # env = Arena2dEnvWrapper(0, True)
    call_back = SaveOnBestTrainingRewardCallback(500, logdir, 1, reward_bound)
    # set temporary model path, if training was interrupted by the keyboard, the current model parameters will be saved.
    path_temp_model = os.path.join(logdir, "DQN_TEMP")
    if not args.restart_training:
        model = DQN(MlpPolicy, env, gamma=GAMMA, learning_rate=LEARNING_RATE,
                    buffer_size=BUFFER_SIZE, target_network_update_freq=SYNC_TARGET_STEPS,tensorboard_log=logdir,verbose=1)
        reset_num_timesteps = True
    else:
        if os.path.exists(path_temp_model+".zip"):
            print("continue training the model...")
            model = DQN.load(path_temp_model, env=env)
            reset_num_timesteps = False
        else:
            print("Can't load the model with the path: {}, please check again!".format(path_temp_model))
            env.close()
            exit(-1)
    # try:
    model.learn(time_steps, log_interval=200, callback=call_back, reset_num_timesteps=reset_num_timesteps)
    model.save(os.path.join(logdir, "DQN_final"))
    # except KeyboardInterrupt:
    #     model.save(path_temp_model)
    #     print("KeyboardInterrupt: saved the current model to {}".format(path_temp_model))
    # finally:
    #     env.close()
    #     exit(0)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restart_training", action="store_true", help="restart the latest unfinished training")
    return parser.parse_args()


if __name__ == "__main__":
    main()
