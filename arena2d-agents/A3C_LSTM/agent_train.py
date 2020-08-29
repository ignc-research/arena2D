import policy_value_lstm
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy
import time

NUM_ACTIONS = 5 # discrete actions

### Hyper Parameters
GAMMA = 0.95			# discount factor
LEARNING_RATE = 0.001	# learning rate for optimizer
ENTROPY_BETA = 0.01		# factor for entropy bonus
REWARD_STEPS = 4		# total number of steps in a chain
CLIP_GRAD = 0.1			# clip gradient to this value to prevent gradients becoming too large
####

AGENT_NAME = "a3c_lstm_agent"

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs > 1)
		# initialize parameters
		self.device = torch.device(device_name)
		self.num_envs = num_envs
		self.num_threads = num_threads
		self.training_data_path = training_data_path

		# initialize network
		self.net = policy_value_lstm.PolicyValueLSTM(num_observations, NUM_ACTIONS)
		self.net.to(self.device)
		if model_name != None:
			self.net.load_state_dict(torch.load(model_name))
		print(self.net)

		# init optimizer
		self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, eps=1e-3)

		# init batches
		self.action_batch = []
		self.reward_batch = []
		self.done_batch = []
		self.state_sequences = []
		self.last_h, self.last_c = self.net.get_initial_hidden(self.device, num=num_envs)
		for i in range(num_envs):
			self.state_sequences.append([])
		self.first_state_idx = [0]*num_envs
	
		# init counter
		self.chain_step = 0
		self.step_count = 0
		self.episode_step = [0]*num_envs
		self.episode_count = 0

		# init metrics
		self.mean_reward = 0.0
		self.mean_success = 0.0
		self.best_reward = -100000.0

		# init timer
		self.start_time = time.time()

		self.last_stats = []

		# constant factor for final reward discount
		self.final_gamma = GAMMA**REWARD_STEPS

				
	def pre_step(self, observations):
		# collecting first state in chain
		if self.chain_step == 0:
			for i in range(self.num_envs):# save index of first state in chain for each environment
				self.first_state_idx[i] = self.episode_step[i]

		# add first observation in episode
		single_sequence = []
		for i in range(self.num_envs):
			single_sequence.append(torch.FloatTensor([observations[i]]).to(self.device))
			if self.episode_step[i] == 0:
				self.state_sequences[i].append(observations[i])

		# pass observations through net, apply softmax to get probability distribution
		single_packed_sequence = pack_sequence(single_sequence)
		policy_v, _, (self.last_h, self.last_c) = self.net(single_packed_sequence, (self.last_h, self.last_c))
		probs = nn.functional.softmax(policy_v, dim=1).data.cpu().numpy()
		# get actions according to probability distribution
		actions = []
		for i in range(self.num_envs):
			a = numpy.random.choice(NUM_ACTIONS, p=probs[i])
			actions.append(a)
		
		self.last_actions = actions
		# return actions
		return actions 

	def post_step(self, new_observations, rewards, dones, mean_reward, mean_success):
		# saving metrics
		self.mean_reward = mean_reward
		self.mean_success = mean_success

		# collecting first action in chain
		if self.chain_step == 0:
			self.action_batch = self.last_actions

		# collecting rewards and dones from all states in chain
		self.reward_batch.extend(rewards)
		self.done_batch.extend(dones)

		# add observation in environment
		for i in range(self.num_envs):
			# add observation only to sequence if episode has not already ended
			if self.chain_step == 0 or self.done_batch[(self.chain_step-1)*self.num_envs + i] == 0:
				self.state_sequences[i].append(new_observations[i])
				self.episode_step[i] += 1

		# update counter
		self.chain_step += 1
		self.step_count += 1

		ret = 0
		#check if chain completed
		if self.chain_step >= REWARD_STEPS:

			# get sequences from start of episode till first state in chain
			# get sequences from first state in chain till end
			first_state_sequences_v = []
			last_state_sequences_v = []
			#print(self.state_sequences[i][:(self.first_state_idx[i]+1)])
			#print(self.state_sequences[i][:(self.first_state_idx[i]+1)])
			for i in range(self.num_envs):
				first_state_sequences_v.append(torch.FloatTensor(self.state_sequences[i][:(self.first_state_idx[i]+1)]).to(self.device))
				last_state_sequences_v.append(torch.FloatTensor(self.state_sequences[i][(self.first_state_idx[i]+1):]).to(self.device))
			first_packed_sequence = pack_sequence(first_state_sequences_v, enforce_sorted=False)
			last_packed_sequence = pack_sequence(last_state_sequences_v, enforce_sorted=False)

			# clear gradients
			self.optimizer.zero_grad()

			# forward first sequences
			policy_v, value_v, hidden_states = self.net(first_packed_sequence, self.net.get_initial_hidden(self.device, self.num_envs))

			# get expected value from last state
			_, last_values_v, _ = self.net(last_packed_sequence, hidden_states)

			# calculate total value from all steps in chain
			total_values = []
			for e in range(self.num_envs):
				total_reward = 0.0
				step_idx = None
				for i in range(REWARD_STEPS):
					step_idx = (REWARD_STEPS-i-1)*self.num_envs+e
					total_reward *= GAMMA 
					total_reward += self.reward_batch[step_idx]
					if self.done_batch[step_idx] == 1: #stop if episode is done
						self.episode_count += 1
						break
				if self.done_batch[step_idx] == 0: # add estimated value for final state if episode is not done
					total_reward += self.final_gamma*last_values_v[e].data.cpu()
						
				total_values.append(total_reward)
			total_values_v = torch.FloatTensor(total_values).to(self.device)

			# calculate value loss
			loss_value_v = nn.functional.mse_loss(value_v.squeeze(-1), total_values_v)
			
			# calculate policy loss 
			log_prob_v = nn.functional.log_softmax(policy_v, dim=1)
			advantage_v = total_values_v - value_v.detach()
			actions_v = torch.LongTensor(self.action_batch).to(self.device)
			log_prob_actions_v = advantage_v * log_prob_v[range(self.num_envs), actions_v]
			loss_policy_v = -log_prob_actions_v.mean()

			# apply softmax and calculate entropy loss
			prob_v = nn.functional.softmax(policy_v, dim=1)
			loss_entropy_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

			# calculate policy gradients
			loss_policy_v.backward(retain_graph=True)
			grads = numpy.concatenate([p.grad.data.cpu().numpy().flatten()
										for p in self.net.parameters() if p.grad is not None])

			# calculate entropy and value gradients
			loss_v = loss_entropy_v + loss_value_v
			loss_v.backward()
			nn.utils.clip_grad_norm_(self.net.parameters(), CLIP_GRAD)
			self.optimizer.step()

			# add policy loss to get total loss
			loss_v += loss_policy_v
			
			# save stats
			self.last_stats = [	("advantage", self.tensorToFloat(advantage_v)),
								("values", self.tensorToFloat(value_v)),
								("batch rewards", float(numpy.mean(total_values))),
								("loss entropy", self.tensorToFloat(loss_entropy_v)),
								("loss policy", self.tensorToFloat(loss_policy_v)),
								("loss value", self.tensorToFloat(loss_value_v)),
								("loss total", self.tensorToFloat(loss_v)),
								("grad l2", float(numpy.sqrt(numpy.mean(numpy.square(grads))))),
								("grad max", float(numpy.max(numpy.abs(grads)))),
								("grad var", float(numpy.var(grads)))]


			# check best mean reward
			if self.mean_reward > self.best_reward and self.episode_count >= 100:
				self.save_model_weights(AGENT_NAME+"_best.dat")
				self.best_reward = self.mean_reward

			# clear batches	
			self.action_batch.clear()
			self.reward_batch.clear()
			self.done_batch.clear()
			self.chain_step = 0

			# get new hidden states
			new_state_sequences = []
			for i in range(self.num_envs):
				new_state_sequences.append(torch.FloatTensor(self.state_sequences[i][:-1]).to(self.device))

			new_packed_sequence = pack_sequence(new_state_sequences, enforce_sorted=False)
			# reset episode step counter if episode done
			self.last_policy, _, (self.last_h, self.last_c) = self.net(new_packed_sequence, self.net.get_initial_hidden(self.device, self.num_envs))
			for i in range(self.num_envs):
				if dones[i] == 1:
					self.episode_step[i] = 0
					self.state_sequences[i].clear()
					# environment will be reset -> reset hidden states
					self.last_h[0,i,:] = 0
					self.last_c[0,i,:] = 0

			ret = 0 # chain complete -> reset environments

		else: # chain still incomplete
			ret = -1


		return ret

	@staticmethod
	def tensorToFloat(tensor):
		return tensor.float().mean().item()

	def save_model_weights(self, filename):
		print("***Saving model weights to %s***"%(filename))
		torch.save(self.net.state_dict(), self.training_data_path + filename)
		with open("%s%s.txt"%(self.training_data_path, filename), "w") as f:
			f.write("Episodes:       %d\n"%(self.episode_count))
			f.write("Steps:        %d\n"%(self.step_count))
			f.write("Mean reward:   %f\n"%(self.mean_reward))
			f.write("Mean success:  %f\n"%(self.mean_success))
			delta_time = int(time.time()-self.start_time);
			f.write("Duration:      %dh %dmin %dsec\n"%(delta_time/3600, (delta_time/60)%60, (delta_time%60)))
			f.write(str(self.net))

	def get_stats(self):
		return self.last_stats

	def stop(self):
		self.save_model_weights(AGENT_NAME+"_final.dat")
		return "Best Mean Reward: " + str(self.best_reward) + "\n\n" + str(self.net)
