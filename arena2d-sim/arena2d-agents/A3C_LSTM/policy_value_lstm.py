import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

HIDDEN_SHAPE_BODY=64
HIDDEN_SHAPE_BODY_OUT=64
HIDDEN_SHAPE_LSTM=32
HIDDEN_SHAPE_POLICY=32
HIDDEN_SHAPE_VALUE=32

class PolicyValueLSTM(nn.Module):
	def __init__(self, num_inputs, num_actions):
		super(PolicyValueLSTM, self).__init__()
		
		self.body = nn.Sequential( 	nn.Linear(num_inputs, HIDDEN_SHAPE_BODY),
									nn.ReLU(),
									nn.Linear(HIDDEN_SHAPE_BODY, HIDDEN_SHAPE_BODY_OUT))

		self.lstm = nn.LSTM(input_size=HIDDEN_SHAPE_BODY_OUT, hidden_size=HIDDEN_SHAPE_LSTM)

		self.policy = nn.Sequential(nn.Linear(HIDDEN_SHAPE_LSTM, HIDDEN_SHAPE_POLICY),
									nn.ReLU(),
									nn.Linear(HIDDEN_SHAPE_POLICY, num_actions))

		self.value = nn.Sequential(	nn.Linear(HIDDEN_SHAPE_LSTM, HIDDEN_SHAPE_VALUE),
									nn.ReLU(),
									nn.Linear(HIDDEN_SHAPE_VALUE, 1))
		
	def forward(self, x:PackedSequence, initial_hidden):
		fc_data = self.body(x.data)
		x = PackedSequence(fc_data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
		_, (h_n, c_n) = self.lstm(x, initial_hidden)
		output = h_n.view(h_n.shape[1], -1)
		policy = self.policy(output)
		value = self.value(output)
		return policy, value, (h_n, c_n)

	def forward_hidden(self, x, hidden):
		num_batches = x.size()[0]
		fc_data = self.body(x)
		fc_data = fc_data.view(1, num_batches, -1)
		_, (h_n, c_n) = self.lstm(fc_data, hidden) 
		output = h_n.view(h_n.shape[1], -1)
		policy = self.policy(output)
		value = self.value(output)
		return policy, value, (h_n, c_n)
	
	def get_initial_hidden(self, device, num=1):
		h = torch.zeros(1, num, HIDDEN_SHAPE_LSTM).to(device)
		c = torch.zeros(1, num, HIDDEN_SHAPE_LSTM).to(device)
		return (h, c)
	
