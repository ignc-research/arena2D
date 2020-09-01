from dqn_models import fc 

import torch
import pathlib

NUM_ACTIONS = 7
STOP_TESTING = 100		# number of episodes used for testing
episode_idx = 0

DEFAULT_MODEL = "../arena2d-sim/dqn_agent_best.dat"

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads,trash):
		global episode_idx
		assert(num_envs == 1)
		if model_name is None:
			model_name=DEFAULT_MODEL
		self.device = torch.device(device_name)
		self.net = fc.FC_DQN(num_observations, NUM_ACTIONS)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		self.net.load_state_dict(torch.load(model_name, map_location=self.device))
		self.net.to(self.device)

		episode_idx = 0

	def pre_step(self, observation):
		# passing observation through net
		state_v = torch.FloatTensor([observation]).to(self.device)
		q_vals_v = self.net(state_v)
		# select action with max q value
		_, act_v = torch.max(q_vals_v, dim=1)
		action = int(act_v.item())

		return action

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success):
	# this function is called after simulation step has been performed
	# return 0 to continue, 1 to stop training
		global episode_idx
		if is_done != 0: # episode done
			episode_idx += 1
			if episode_idx >= STOP_TESTING:
				return 1 #stop testing
		return 0 # keep on testing

	def get_stats(self):
		return [("Epsilon", self.epsilon),
				("Mean Value", self.mean_value),
				("Mean Loss", self.mean_loss)]

	def stop(self):
		pass	
