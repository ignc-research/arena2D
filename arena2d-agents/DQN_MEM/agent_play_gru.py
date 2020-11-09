from dqn_models import gru
from torch.nn.utils.rnn import pack_sequence
import torch
NUM_ACTIONS = 5
num_observations=362
additional_state=2
BATCH_SIZE = 64
USE_GRU = True
SEQ_LENGTH=64
SEQ_LENGTH_MAX=30000
DISCOUNT_FACTOR=0.99

#model_name="../arena2d-agents/DQN/dqn_agent_Gtr_best.dat"/home/junhui/study/VIS/Arena_work2/arena2d-sim/arena2d-agents/agent_dqn_sequence
class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)

		self.device = torch.device(device_name)
		self.reset()
		self.model_name="/home/junhui/study/VIS/Arena_work2/arena2d-sim/arena2d-agents/agent_dqn_sequence/dqn_agent_best_gru_l2_64.dat"  # need to be modified for different models
		#self.tensor_step_buffer[self.episode_idx]=int(self.episode_frames)
		self.net = gru.GRUModel(num_observations+additional_state, NUM_ACTIONS)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		self.net.to(self.device)
		self.h = self.net.init_hidden(1).data		
		if self.model_name != None:
		        print("model_name:",self.model_name)
		        self.net.load_state_dict(torch.load(self.model_name, map_location=self.device));                       			
	def pre_step(self, observation):
		# passing observation through net
		q = None
		self.tensor_state_buffer[self.episode_idx] = torch.FloatTensor([self.last_reward,self.last_action]+observation);
		if self.episode_idx > SEQ_LENGTH-1:
			start_index= self.episode_idx-(SEQ_LENGTH-1)
			L=SEQ_LENGTH
		else:
			start_index = 0
			L=self.episode_idx+1
		state_v=[torch.narrow(self.tensor_state_buffer, dim=0, start=start_index, length=L)]
		t=pack_sequence(state_v, enforce_sorted=False)
		q,self.h = self.net(t,self.h)
		q=q.view(-1,NUM_ACTIONS)		
		# select action with max q value
		_, act_v = torch.max(q, dim=1)
		action = int(act_v.item())
		self.last_action = action
		return action

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success,mean_collision):
	        self.last_reward = reward*DISCOUNT_FACTOR
	        if is_done:		
		         self.reset()
	        return 0

		
					
	def stop(self):
		pass

	def reset(self):
		self.episode_idx=0
		self.last_action=-1
		self.last_reward=0.0
		self.tensor_state_buffer = torch.zeros(SEQ_LENGTH_MAX, num_observations+additional_state,dtype=torch.float).to(self.device)
