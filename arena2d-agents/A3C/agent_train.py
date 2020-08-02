import policy_value_fc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy
import time

NUM_ACTIONS = 6 # discrete actions

### Hyper Parameters
GAMMA = 0.95			# discount factor
LEARNING_RATE = 0.001	# learning rate for optimizer
ENTROPY_BETA = 0.01		# factor for entropy bonus
REWARD_STEPS = 4		# total number of steps in a chain
CLIP_GRAD = 0.1			# clip gradient to this value to prevent gradients becoming too large
####

AGENT_NAME = "a3c_agent"

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs > 1)
		# initialize parameters
		self.device = torch.device(device_name)
		self.num_envs = num_envs
		self.num_threads = num_threads
		self.training_data_path = training_data_path

		# initialize network
		self.net = policy_value_fc.PolicyValueFC(num_observations, NUM_ACTIONS)
		self.net.to(self.device)
		if model_name != None:
			self.net.load_state_dict(torch.load(model_name))
		print(self.net)

		# init optimizer
		self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, eps=1e-3)

		# init batches
		self.state_batch = []
		self.action_batch = []
		self.reward_batch = []
		self.done_batch = []

		# init counter
		self.chain_step = 0
		self.step_count = 0
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
			self.state_batch = observations

		# pass observations through net, apply softmax to get probability distribution
		obs_v = torch.FloatTensor(observations).to(self.device)
		policy_v = self.net.get_policy(obs_v)
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

		self.chain_step += 1
		self.step_count += 1

		#check if chain completed
		if self.chain_step >= REWARD_STEPS:
			# get expected value from last state
			last_states_v = torch.FloatTensor(new_observations).to(self.device)
			last_values_v = self.net.get_value(last_states_v)
			
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

			# clear gradients
			self.optimizer.zero_grad()

			# forward first state in chain and calculate loss
			states_v = torch.FloatTensor(self.state_batch).to(self.device)
			policy_v, value_v = self.net(states_v)

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
			self.state_batch.clear()
			self.action_batch.clear()
			self.reward_batch.clear()
			self.done_batch.clear()
			self.chain_step = 0

			return 0 # chain complete -> reset environments

		else: # chain still incomplete
			return -1 # do not reset environments 

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
