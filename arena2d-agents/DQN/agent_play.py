from dqn_models import fc 

import torch
import pathlib

NUM_ACTIONS = 7

DEFAULT_MODEL = "../arena2d-agents/DQN/best.dat"

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads):
		assert(num_envs == 1)
		if model_name is None:
			model_name=DEFAULT_MODEL
		self.device = torch.device(device_name)
		self.net = fc.FC_DQN(num_observations, NUM_ACTIONS)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		self.net.load_state_dict(torch.load(model_name, map_location=self.device))
		self.net.to(self.device)

	def pre_step(self, observation):
		# passing observation through net
		state_v = torch.FloatTensor([observation]).to(self.device)
		q_vals_v = self.net(state_v)
		# select action with max q value
		_, act_v = torch.max(q_vals_v, dim=1)
		action = int(act_v.item())
		# swap backward and stop action (changed in arena2d since training)
		if action == 6:
			action = 5
		elif action == 5:
			action = 6
		return action

	def post_step(new_observation, action, reward, mean_reward, is_done, mean_success):
		return 0 # keep on playing

	def stop(self):
		pass	
