from dqn_models import Gtr
from torch.nn.utils.rnn import pad_sequence
import torch
NUM_ACTIONS = 5
num_observations=362

SEQ_LENGTH=64
SEQ_LENGTH=30000

#model_name="../arena2d-agents/DQN/dqn_agent_Gtr_best.dat"/home/junhui/study/VIS/Arena_work2/arena2d-sim/arena2d-agents/agent_dqn_sequence
class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)

		self.device = torch.device(device_name)
		self.reset()
		self.model_name="/home/junhui/study/VIS/Arena_work2/arena2d-sim/arena2d-agents/agent_dqn_sequence/dqn_agent_best_gtr_3_layer_l2.dat"
		#self.tensor_step_buffer[self.episode_idx]=int(self.episode_frames)
		self.net = Gtr.TransformerDqn(NUM_ACTIONS,num_observations)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		self.net.to(self.device)
		
		if self.model_name != None:
		        print("model_name:",self.model_name)
		        self.net.load_state_dict(torch.load(self.model_name, map_location=self.device));                       			
	def pre_step(self, observation):
		# passing observation through net
		q = None
		self.tensor_state_buffer[self.episode_idx] = torch.FloatTensor(observation);
		if self.episode_idx > SEQ_LENGTH-1:
			start_index= self.episode_idx-(SEQ_LENGTH-1)
			L=SEQ_LENGTH
		else:
			start_index = 0
			L=self.episode_idx+1
		state_v=[torch.narrow(self.tensor_state_buffer, dim=0, start=start_index, length=L)]
		#state_v=[torch.narrow(self.tensor_state_buffer, dim=0, start=0, length=self.episode_idx+1)]
		t=pad_sequence(state_v).data.to(self.device)
		q = self.net(t)
		q=q.view(-1,NUM_ACTIONS)		
		# select action with max q value
		_, act_v = torch.max(q, dim=1)
		action = int(act_v.item())
		return action

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success,mean_collision):
		if is_done:
			self.reset()
		return 0 # keep on playing
	
	def get_stats(self):
		return []

	def stop(self):
		pass

	def reset(self):
		self.episode_idx=0
		self.tensor_state_buffer = torch.zeros(SEQ_LENGTH, num_observations ,dtype=torch.float).to(self.device)
