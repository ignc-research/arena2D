import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_sequence
import transformer
import numpy
import random
from collections import deque

### hyper parameters ###
MEAN_REWARD_BOUND = 120.0	# training is considered to be done if the mean reward reaches this value
NUM_ACTIONS = 6				# total number of discrete actions the robot can perform
DISCOUNT_FACTOR = 0.99		# discount factor for reward estimation (often denoted by gamma)
SYNC_TARGET_STEPS = 2000	# target net is synchronized with net every X steps
LEARNING_RATE = 0.00025 	# learning rate for optimizer
EPSILON_START = 1			# start value of epsilon
EPSILON_MAX_STEPS = 10**6	# how many steps until epsilon reaches minimum
EPSILON_END = 0.02			# min epsilon value
BATCH_SIZE = 16				# batch size for training after every step
TRAINING_START = 10000		# start training only after the first X steps
MEMORY_SIZE = 2**20			# last X states will be stored in a buffer (memory), from which the batches are sampled
N_STEPS = 2					# number of steps when evaluating bellman equation
DOUBLE = True				# activate double DQN
TRANSFORMER_TYPE = "id"	# one of "base": standard transformer, "id": identity map reorder, "gated": gated transformer
TRANSFORMER_NUM_LAYERS = 4	# number of encoder layers in transformer
TRANSFORMER_NUM_HEADS = 8	# number of heads for multi-head self attention, make sure num_observations is divisible by this number
TRANSFORMER_FF_DIM = 2048	# number of neurons in feedforward layer of transformer
TRANSFORMER_DROPOUT = 0.1	# dropout applied after each transformer submodule
#######################

AGENT_NAME="dqn_transformer_agent"

class QTransformer(nn.Module):
	def __init__(self, num_observations, device):
		super(QTransformer, self).__init__()

		# positional encoding
		self.pos_encoder = transformer.PositionalEncoding(num_observations)

		# create transformer with N layers
		self.transformer = transformer.Transformer(num_observations,
											nhead = TRANSFORMER_NUM_HEADS,
											num_layers = TRANSFORMER_NUM_LAYERS,
											transformer_type = TRANSFORMER_TYPE,
											dim_feedforward = TRANSFORMER_FF_DIM,
											dropout = TRANSFORMER_DROPOUT)

		# map output of final layer to Q values of actions
		self.linear = nn.Linear(num_observations, NUM_ACTIONS)
		
		# save device for forwarding
		self.device = device

	# x is a list of N sequences or a tensor (sequence, batch, observation)
	# returns a tensor (N, NUM_ACTIONS) containing q values for 
	def forward(self, x):
		x_transform = None
		# sequences with different lengths
		if type(x) == list:
			x_transform = torch.Tensor(len(x), x[0].size()[1]).to(self.device)
			# forward each sequence in batch through transformer separately because of different sequence length
			for i in range(len(x)):
				seq_len = x[i].size()[0]
				x_encoded = self.pos_encoder(x[i])
				x_transform[i] = self.transformer(x_encoded.view(seq_len, 1, -1))[-1]
		else: # sequences of same length -> tensor
			x_transform = self.transformer(self.pos_encoder(x))[-1]


		# feed batch through linear
		q_vals = self.linear(x_transform)
		return q_vals
	

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)
		self.device = torch.device(device_name)
		self.training_data_path = training_data_path
		self.num_observations = num_observations

		# creating xp buffers on gpu for faster sampling
		self.tensor_state_buffer = torch.zeros(MEMORY_SIZE, num_observations ,dtype=torch.float).to(self.device)# state
		# rewards with applied N-Step: buffer[t] = reward_t + discount*buffer[t+1] + ... discount^(N-1)*buffer[t+N-1]
		self.tensor_reward_buffer = torch.zeros(MEMORY_SIZE, dtype=torch.float).to(self.device)
		self.tensor_action_buffer = torch.zeros(MEMORY_SIZE, dtype=torch.long).to(self.device)# the action that was chosen
		self.tensor_done_buffer = torch.zeros(MEMORY_SIZE, dtype=torch.bool).to(self.device)# episode has ended
		self.tensor_step_buffer = torch.zeros(MEMORY_SIZE, dtype=torch.int16).cpu()# step index in episode (starting at 0)

		# creating net and target net
		self.net = QTransformer(num_observations, self.device)
		self.tgt_net = QTransformer(num_observations, self.device)


		# copy to device
		self.net.to(self.device)
		self.tgt_net.to(self.device)

		# showing net to user
		print(self.net)

		self.episode_start = 0

		# load model parameters from file if given
		if model_name is not None:
			self.net.load_state_dict(torch.load(model_name))
			self.tgt_net.load_state_dict(self.net.state_dict())

		# initialize epsilon for epsilon-greedy algorithm
		self.epsilon = EPSILON_START

		# create optimizer
		self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

		# mean rewards
		self.best_mean_reward = None
		self.mean_reward = 0

		self.new_state_discount = DISCOUNT_FACTOR**N_STEPS
		# time metrics
		self.sampling_times = deque(maxlen=100)
		self.batch_forward_times =deque(maxlen=100)
		self.loss_calc_times = deque(maxlen=100)
		self.backward_times = deque(maxlen=100)
		self.optimize_times = deque(maxlen=100)
		self.gpu_pre_copy_times = deque(maxlen=100)
		self.gpu_copy_times = deque(maxlen=100)
		self.measure_gpu_times = False

		# training metrics
		self.mean_loss_buffer = deque(maxlen=100)
		self.mean_value_buffer = deque(maxlen=100)
		self.mean_loss = 0
		self.mean_value = 0
		self.mean_seq_len = deque(maxlen=100)
		self.mean_seq_len.append(0)
		self.mean_seq_len_value = 0

		# initializing frame indicies
		self.frame_idx = 0
		self.last_episode_frame = 0
		self.episode_idx = self.episode_start
		self.training_done = False

		# getting current time
		self.last_time = time.perf_counter()
		self.start_time = time.time();

		# reset state
		self.reset()

	# calculate epsilon according to current frame index
	def epsilon_decay(self):
		self.epsilon = max(EPSILON_END, EPSILON_START - self.frame_idx/float(EPSILON_MAX_STEPS))

	# reset collected reward
	def reset(self):
		self.total_reward = 0.0
		self.episode_frames = 0
		self.last_observation = None
	
	def pre_step(self, observation):
		# measuring gpu times only every 100 frames (better performance)
		self.measure_gpu_times = (self.frame_idx%100 == 0)
		self.last_observation = observation
		self.epsilon_decay()
		self.start_gpu_measure()
		action = -1
		# insert current state into buffer
		idx = self.frame_idx%MEMORY_SIZE
		self.tensor_state_buffer[idx] = torch.FloatTensor(observation);
		self.tensor_step_buffer[idx] = self.episode_frames
		self.stop_gpu_measure(self.gpu_pre_copy_times)
		if random.random() <= self.epsilon: # random action
			action = random.randrange(0, NUM_ACTIONS)
		else:
			# pack sequence
			sequence = Agent.swap_batch_seq(self.get_complete_sequence(idx).unsqueeze(0))
			q = self.net(sequence)
			max_value, act_v = torch.max(q, dim=1)
			self.mean_value_buffer.append(max_value.item())
			action = int(act_v.item())

		self.last_action = action
		return action
	
	# get episode sequence from replay buffer
	def get_sequence(self, start_index, seq_len):
		sequence = None
		if start_index+seq_len > MEMORY_SIZE: # wrap around -> two part sequence
			# sequence part 1
			seq1_len = MEMORY_SIZE-start_index
			seq1 = torch.narrow(self.tensor_state_buffer, dim=0, start=start_index, length=seq1_len)
			# sequence part 2
			seq2_len = seq_len - seq1_len
			seq2 = torch.narrow(self.tensor_state_buffer, dim=0, start=0, length=seq2_len)
			sequence = torch.cat((seq1, seq2), 0)
		else:# continuous sequence 
			sequence = torch.narrow(self.tensor_state_buffer, dim=0, start=start_index, length=seq_len)

		return sequence
	
	def get_complete_sequence(self, end_index):
		seq_len = int(self.tensor_step_buffer[end_index])
		return self.get_sequence((end_index+(MEMORY_SIZE-seq_len))%MEMORY_SIZE, seq_len+1)

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success):
		self.start_gpu_measure()
		idx = self.frame_idx%MEMORY_SIZE
		if is_done: # save next state if done, because next pre_step will have different state
			self.tensor_state_buffer[(idx+1)%MEMORY_SIZE] = torch.FloatTensor(new_observation)
		self.tensor_reward_buffer[idx] = reward
		self.tensor_action_buffer[idx] = self.last_action
		self.tensor_done_buffer[idx] = (is_done != 0)
		# update reward from last n steps
		max_steps = min(N_STEPS, self.episode_frames+1)
		discount = 1
		for i in range(1, max_steps):
			discount *= DISCOUNT_FACTOR
			pre_idx = (MEMORY_SIZE+idx-i)%MEMORY_SIZE
			self.tensor_reward_buffer[pre_idx] += reward * discount
		# set done of last n steps to true if episode has ended
		if is_done != 0:
			for i in range(1, max_steps):
				pre_idx = (MEMORY_SIZE+idx-i)%MEMORY_SIZE
				self.tensor_done_buffer[pre_idx] = True

		# stop gpu measure
		self.stop_gpu_measure(self.gpu_copy_times)

		# calculate metrics
		self.mean_reward = mean_reward
		self.total_reward += reward
		self.mean_success = mean_success
		if len(self.mean_value_buffer) > 0:
			self.mean_value = numpy.mean(list(self.mean_value_buffer))
		if len(self.mean_loss_buffer) > 0:
			self.mean_loss = numpy.mean(list(self.mean_loss_buffer))

		# count total frames/episode frames
		self.frame_idx += 1
		self.episode_frames += 1

		if is_done != 0: # episode done
			self.episode_idx += 1 # increment episode count

			# check if new best mean
			self.check_best_mean()
			
			self.mean_seq_len_value = numpy.mean(list(self.mean_seq_len))
			# writing metrics
			self.write_stats()

			# check whether mean reward bound reached
			if mean_reward >= MEAN_REWARD_BOUND and (self.episode_idx-self.episode_start) >= 100:
				print("Solved in %d episodes (%d frames)!"%(self.episode_idx, self.frame_idx))
				self.save_model_weights("%s_final.dat"%(AGENT_NAME));
				self.training_done = True
				return 1 # stop training 

		# only optimize if enough data in buffer
		if self.frame_idx >= TRAINING_START:
			# set weights in target net from training net every SYNC_TARGET_STEPS frames
			if self.frame_idx % SYNC_TARGET_STEPS == 0:
				self.tgt_net.load_state_dict(self.net.state_dict())

			# optimizing weights through SGD
			self.optimizer.zero_grad()
			self.start_gpu_measure()
			batch = self.sample(BATCH_SIZE)
			self.stop_gpu_measure(self.sampling_times)

			loss_t = self.calc_loss(*batch)
			
			self.start_gpu_measure()
			loss_t.backward()
			self.stop_gpu_measure(self.backward_times)

			self.start_gpu_measure()
			t_before = time.perf_counter()
			self.optimizer.step()
			self.stop_gpu_measure(self.optimize_times)

		if is_done != 0:
			# reset episode
			self.reset()

		return 0 # continue training

	def write_stats(self):
		# calculate and write metrix
		speed = (self.frame_idx - self.last_episode_frame)/(time.perf_counter() - self.last_time)
		self.last_episode_frame = self.frame_idx
		mean_sampling = "-"
		mean_loss_calc = "-"
		mean_optimize = "-"
		mean_batch = "-"
		mean_backward = "-"
		mean_gpu_copy = "-"
		best_reward = "-"
		memory_size = min(self.frame_idx, MEMORY_SIZE)
		if len(self.gpu_copy_times) > 0:
			mean_gpu_copy = "%.3fms"%((numpy.mean(list(self.gpu_copy_times)) + numpy.mean(list(self.gpu_pre_copy_times)))*1000)
		if len(self.loss_calc_times) > 0:
			mean_loss_calc = "%.3fms"%(numpy.mean(list(self.loss_calc_times))*1000)
		if len(self.optimize_times) > 0:
			mean_optimize = "%.3fms"%(numpy.mean(list(self.optimize_times))*1000)
		if len(self.sampling_times) > 0:
			mean_sampling = "%.3fms"%(numpy.mean(list(self.sampling_times))*1000)
		if len(self.batch_forward_times) > 0:
			mean_batch = "%.3fms"%(numpy.mean(list(self.batch_forward_times))*1000)
		if len(self.backward_times) > 0:
			mean_backward = "%.3fms"%(numpy.mean(list(self.backward_times))*1000)
		if self.best_mean_reward is not None:
			best_reward = "%.3f"%(self.best_mean_reward)
		print("  ----- Agent -----")
		print("  Times:")
		print("   -> GPU copy:         "+mean_gpu_copy)
		print("   -> Batch sampling:   "+mean_sampling)
		print("   -> Batch forwarding: "+mean_batch)
		print("   -> Loss calculation: "+mean_loss_calc)
		print("   -> Loss backward:    "+mean_backward)
		print("   -> Optimizing:       "+mean_optimize)
		print("  Best reward: "+best_reward)
		print("  Epsilon:      %.2f"%(self.epsilon))
		print("  Frames:       %d"%(self.frame_idx))
		print("  Memory:       %.1f%% (%d/%d)"%(float(100*memory_size)/float(MEMORY_SIZE), memory_size, MEMORY_SIZE))
		print("  Mean loss:    %.3f"%(self.mean_loss))
		print("  Mean value:   %.3f"%(self.mean_value))
		print("  Mean Seq-Len: %.1f"%(self.mean_seq_len_value))
		# reset timer
		self.last_time = time.perf_counter()
		
	
	# check if new best mean and if reached boundary
	def check_best_mean(self):
		if (self.best_mean_reward is None or self.best_mean_reward < self.mean_reward) and (self.episode_idx-self.episode_start) >= 100:
			self.save_model_weights(AGENT_NAME + "_best.dat")
			self.best_mean_reward = self.mean_reward
	
	def save_model_weights(self, filename):
		print("***Saving model weights to %s***"%(filename))
		torch.save(self.net.state_dict(), self.training_data_path+filename)
		with open("%s%s.txt"%(self.training_data_path, filename), "w") as f:
			f.write("Episode:       %d\n"%(self.episode_idx))
			f.write("Start episode: %d\n"%(self.episode_start))
			f.write("Frames:        %d\n"%(self.frame_idx))
			f.write("Mean reward:   %f\n"%(self.mean_reward))
			f.write("Mean success:  %f\n"%(self.mean_success))
			f.write("Epsilon:       %f\n"%(self.epsilon))
			delta_time = int(time.time()-self.start_time);
			f.write("Duration:      %dh %dmin %dsec\n"%(delta_time/3600, (delta_time/60)%60, (delta_time%60)))
			f.write(str(self.net))
	
	def start_gpu_measure(self):
		if self.measure_gpu_times:
			if self.tensor_state_buffer.is_cuda:
				torch.cuda.synchronize(self.device)
			self.time_before = time.perf_counter()

	def stop_gpu_measure(self, mean_buffer):
		if self.measure_gpu_times:
			if self.tensor_state_buffer.is_cuda:
				torch.cuda.synchronize(self.device)
			time_after = time.perf_counter()
			mean_buffer.append(time_after-self.time_before)

	# returns batch of size @batch_size as tuple with random samples from xp buffers
	# (state, new_state, action, reward, is_done)
	def sample(self, batch_size):
		# sample random elements
		r = min(self.frame_idx-N_STEPS, MEMORY_SIZE)
		next_random_indicies = numpy.random.choice(r, batch_size)

		# buffer wraps around
		if self.frame_idx > MEMORY_SIZE:
			f = self.frame_idx%MEMORY_SIZE
			for i in range(len(next_random_indicies)):
				i_next = next_random_indicies[i]
				i_first = i_next - int(self.tensor_step_buffer[i_next])
				i_border = (MEMORY_SIZE+i_first)%MEMORY_SIZE
				if i_border < f:
					next_random_indicies[i] = i_first 
					
		# sample random state indicies
		current_idx = self.frame_idx%MEMORY_SIZE
		random_indicies = numpy.empty(batch_size, dtype=numpy.long)
		shortest_seq_len = 100000
		for i in range(len(random_indicies)):
			next_index = next_random_indicies[i]
			seq_len = int(self.tensor_step_buffer[next_index])+1
			# episode is currently overwritten through ring buffer
			# find shortest sequence
			if  seq_len <= N_STEPS: # sample is too close at beginning of episode
				next_random_indicies[i] = (next_index+1+(N_STEPS-seq_len))%MEMORY_SIZE
				next_index = next_random_indicies[i]
				seq_len = N_STEPS+1

			shortest_seq_len = min(shortest_seq_len, seq_len)
			random_indicies[i] = (next_index+(MEMORY_SIZE-N_STEPS))%MEMORY_SIZE
	
		self.mean_seq_len.append(shortest_seq_len)
		t_seq_next = torch.Tensor(len(next_random_indicies), shortest_seq_len, self.num_observations).to(self.device)
		t_seq = torch.Tensor(len(next_random_indicies), shortest_seq_len-N_STEPS, self.num_observations).to(self.device)
		for i in range(len(next_random_indicies)):
			t_seq_next[i] = self.get_sequence((next_random_indicies[i]+(MEMORY_SIZE-shortest_seq_len+1))%MEMORY_SIZE, shortest_seq_len)
			t_seq[i] = torch.narrow(t_seq_next[i], dim=0, start=0, length=shortest_seq_len-N_STEPS)
		# copy indicies to gpu for faster sampling
		random_indicies_v = torch.tensor(random_indicies).to(self.device)
		# get actual tensors from indicies
		action = self.tensor_action_buffer[random_indicies_v]
		reward = self.tensor_reward_buffer[random_indicies_v]
		is_done = self.tensor_done_buffer[random_indicies_v]
		return (Agent.swap_batch_seq(t_seq), Agent.swap_batch_seq(t_seq_next), action, reward, is_done)

	# swapping axis for batch and sequence in tensor passed to transformer
	def swap_batch_seq(t):
		return t.permute(1, 0, 2)

	def calc_loss(self, states, next_states, actions, rewards, dones):
		self.start_gpu_measure()
		# get currently estimated action values of performed actions in the past
		q_values = self.net(states);
		state_action_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
		# estimate action values of next states
		if DOUBLE:# double dqn
			next_state_actions = self.net(next_states).max(1)[1]
			next_state_values = self.tgt_net(next_states).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
		else:
			next_state_values = self.tgt_net(next_states).max(1)[0]
		# set estimated state values of done states to 0
		next_state_values[dones] = 0.0
		# detatch from flow graph
		next_state_values = next_state_values.detach()
		# Bellman
		expected_state_action_values = next_state_values * self.new_state_discount + rewards
		self.stop_gpu_measure(self.batch_forward_times)

		# loss function
		self.start_gpu_measure()
		l = nn.MSELoss()(state_action_values, expected_state_action_values)
		self.stop_gpu_measure(self.loss_calc_times)
		self.mean_loss_buffer.append(l.item())
		return l
	
	def get_stats(self):
		return [("Epsilon", self.epsilon),
				("Mean Value", self.mean_value),
				("Mean Loss", self.mean_loss),
				("Mean Sequence Length", self.mean_seq_len_value)]

	def stop(self):
		if not self.training_done:
			self.save_model_weights("%s_episode%d.dat"%(AGENT_NAME, self.episode_idx))

		return_str = "Best Mean Reward: "
		if self.best_mean_reward is None:
			return_str += "-"
		else:
			return_str +=str(self.best_mean_reward)

		return_str += "\n\n"
		return_str += str(self.net)

		return return_str
