import rospy
from stable_baselines.common.vec_env import SubprocVecEnv
from rl_ros_agents.env_wappers.arena2dEnv import get_arena_envs,Arena2dEnvWrapper
from stable_baselines import A2C
from stable_baselines.common.policies import MlpLstmPolicy
import tensorflow as tf
# disable tensorflow deprecated information
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    env = get_arena_envs()

    model = A2C(MlpLstmPolicy, env, verbose=1)
    model.learn(total_timesteps=int(1e6))
    model.save("a2c_arena_env_4")
    
