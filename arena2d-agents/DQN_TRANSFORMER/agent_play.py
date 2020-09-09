import time
import torch
import transformer
import numpy
import random

NUM_ACTIONS = 6

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)
		assert(model_name is not None)
		self.device = torch.device(device_name)
		self.num_observations = num_observations

		# creating net
		self.net = transformer.QTransformer(num_observations, NUM_ACTIONS, self.device)

		# copy to device
		self.net.to(self.device)

		# showing net to user
		print(self.net)

		# load model parameters from file if given
		self.net.load_state_dict(torch.load(model_name, map_location=self.device))
		self.net.train(False)

		# sequence of observations
		self.sequence = []

	def pre_step(self, observation):
		# add to sequence
		self.sequence.append(observation)
		
		# forward through net
		q = self.net(torch.Tensor(self.sequence).unsqueeze(1).to(self.device))
		_, act_v = torch.max(q, dim=1)
		action = int(act_v.item())

		return action
	
	def get_stats(self):
		pass

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success):
		if is_done == 1:
			self.sequence.clear()
		return 0 # continue playing

	def stop(self):
		pass
