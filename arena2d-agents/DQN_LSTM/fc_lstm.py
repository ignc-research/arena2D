import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

HIDDEN_SHAPE_FIRST = 64
HIDDEN_SHAPE_SECOND = 32
HIDDEN_SHAPE_LSTM = 32
# fully connected dqn
class FC_LSTM(nn.Module):
	def __init__(self, input_shape, num_actions):
		super(FC_LSTM, self).__init__()

		# first fully connected layers
		self.fc = nn.Sequential(nn.Linear(input_shape, HIDDEN_SHAPE_FIRST),
								nn.ReLU(),
								nn.Linear(HIDDEN_SHAPE_FIRST, HIDDEN_SHAPE_SECOND))
		self.lstm = nn.LSTM(input_size=HIDDEN_SHAPE_SECOND, hidden_size=HIDDEN_SHAPE_LSTM)
		self.fc2 =nn.Sequential(nn.Linear(HIDDEN_SHAPE_LSTM, num_actions))

	def forward(self, x:PackedSequence):
		fc_data = self.fc(x.data)
		x = PackedSequence(fc_data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
		_, (h_n, _) = self.lstm(x) # feed through lstm with initial hidden state = zeros
		output = h_n.view(h_n.shape[1], -1)
		output = self.fc2(output)
		return output
	
	def forward_hidden(self, x, hidden):
		fc_data =  self.fc(x)
		fc_data = fc_data.view(1, 1, -1)
		_, (h_n, c_n) = self.lstm(fc_data, hidden)
		output = h_n.view(h_n.shape[1], -1)
		output = self.fc2(output)
		return output, (h_n, c_n)

	def getInitialHidden(self):
		h = torch.zeros(1, 1, HIDDEN_SHAPE_LSTM)
		c = torch.zeros(1, 1, HIDDEN_SHAPE_LSTM)
		return (h, c)
