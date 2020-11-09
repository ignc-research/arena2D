from dqn_models import fc 

import torch
import pathlib

NUM_ACTIONS = 5

DEFAULT_MODEL = "../arena2d-agents/DQN/best.dat"

class Agent:
	def __init__(self, device, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)
		self.model_name="/home/junhui/study/VIS/Arena_work2/arena2d-sim/arena2d-agents/agent_dqn_sequence/dqn_agent_best_fc_l3.dat"
		self.device = torch.device(device)
		self.net = fc.FC_DQN(num_observations, NUM_ACTIONS)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		self.net.load_state_dict(torch.load(self.model_name, map_location=self.device))
		self.net.to(self.device)

	def pre_step(self, observation):
		# passing observation through net
		state_v = torch.FloatTensor([observation]).to(self.device)
		q_vals_v = self.net(state_v)
		# select action with max q value
		_, act_v = torch.max(q_vals_v, dim=1)
		action = int(act_v.item())
		return action

	def post_step(new_observation, action, reward, mean_reward, is_done, mean_success,mean_collision):
		return 0 # keep on playing

	def stop(self):
		pass	
