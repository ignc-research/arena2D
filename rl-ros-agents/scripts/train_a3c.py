import rospy
from stable_baselines.common.vec_env import SubprocVecEnv
from arena2d_ros_agents.env_wrapper.arena2dEnv import Arena2dEnvWrapper

def get_envs(num_envs):
    env = SubprocVecEnv([lambda i=i: Arena2dEnvWrapper(i) for i in range(num_envs)])
    a = 1

def main():
    pass

if __name__ == "__main__":
    get_envs(4)
    # rospy.init_node("haha")
    