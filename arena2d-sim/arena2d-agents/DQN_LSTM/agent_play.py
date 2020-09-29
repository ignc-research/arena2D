import fc_lstm
import fc_gru
import torch

NUM_ACTIONS = 6
USE_GRU = True

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)
		self.device = torch.device(device_name)
		if USE_GRU:
			self.net = fc_gru.FC_GRU(num_observations, NUM_ACTIONS)
		else:
			self.net = fc_lstm.FC_LSTM(num_observations, NUM_ACTIONS)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		self.net.to(self.device)
		if model_name != None:
			self.net.load_state_dict(torch.load(model_name, map_location=device))
		self.reset_hidden()

	def pre_step(self, observation):
		# passing observation through net
		q = None
		t = torch.FloatTensor([observation]).to(self.device)
		if USE_GRU:
			q, self.last_h = self.net.forward_hidden(t, self.last_h)
		else:
			q, (self.last_h, self.last_c) = self.net.forward_hidden(t, (self.last_h, self.last_c))
		# select action with max q value
		_, act_v = torch.max(q, dim=1)
		action = int(act_v.item())
		return action

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success):
		if is_done:
			self.reset_hidden()
		return 0 # keep on playing
	
	def get_stats(self):
		return []

	def stop(self):
		pass

	def reset_hidden(self):
		if USE_GRU:
			self.last_h = self.net.getInitialHidden();
			self.last_h = self.last_h.to(self.device)
		else:
			(self.last_h, self.last_c) = self.net.getInitialHidden();
			self.last_h = self.last_h.to(self.device)
			self.last_c = self.last_c.to(self.device)
