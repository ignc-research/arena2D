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
# disable tensorflow deprecated information
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

LOGDIR = None
GAMMA = 0.95
LEARNING_RATE = 1e-3
N_STEPS = 4
MAX_GRAD_NORM = 0.1

TIME_STEPS = int(1e10)
REWARD_BOUND = 100
use_reward_bound = True


def main():
    if LOGDIR is None:
        defaul_log_dir = os.path.join("results", "A2C_" + getTimeStr())
        os.makedirs(defaul_log_dir, exist_ok=True)
        logdir = defaul_log_dir
    else:
        logdir = LOGDIR
    if use_reward_bound:
        time_steps = sys.maxsize  # set this to inf
        reward_bound = REWARD_BOUND
    else:
        time_steps = TIME_STEPS
        reward_bound = None

    envs = get_arena_envs(log_dir=logdir)
    call_back = SaveOnBestTrainingRewardCallback(100, logdir, 1, reward_bound)
    model = A2C(MlpLstmPolicy, envs, verbose=1, n_steps=N_STEPS, max_grad_norm=MAX_GRAD_NORM,
                learning_rate=LEARNING_RATE, gamma=GAMMA, tensorboard_log=logdir)
    model.learn(time_steps, log_interval=50, callback=call_back)
    model.save(os.path.join(logdir, "A2C_final"))
    envs.close()


if __name__ == "__main__":
    main()
