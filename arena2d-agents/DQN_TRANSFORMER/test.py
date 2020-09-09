import torch
import agent_train
import random

NUM_OBSERVATIONS = 8

def get_observation():
	return [random.random() for x in range(8)]

if __name__ == '__main__':
	device_name = "cpu"
	device = torch.device(device_name)

	a = agent_train.Agent(device_name=device_name, model_name=None, num_observations=NUM_OBSERVATIONS, num_envs=1, num_threads=1, training_data_path = "./")
	seq = [torch.randn(5, NUM_OBSERVATIONS).to(device), torch.randn(3, 8).to(device), torch.randn(10, 8).to(device), torch.randn(5, 8).to(device), torch.randn(3, 8).to(device), torch.randn(10, 8).to(device), torch.randn(3, 8).to(device), torch.randn(10, 8).to(device)]
	obs = get_observation()
	for i in range(10):
		a.pre_step(obs)
		obs = get_observation()
		a.post_step(obs, 0, 0, 0, 0)

	a.pre_step(obs)
	a.post_step(get_observation(), 0, 1, 0, 0)
	obs = get_observation()
	for i in range(20):
		a.pre_step(obs)
		obs = get_observation()
		a.post_step(obs, 0, 0, 0, 0)
	print(a.tensor_state_buffer)
	print(a.tensor_step_buffer)
	print(str(a.sample(50)))
	print(a.get_complete_sequence(12))

