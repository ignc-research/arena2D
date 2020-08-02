import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

HIDDEN_SHAPE_FIRST = 64
HIDDEN_SHAPE_SECOND = 32
HIDDEN_SHAPE_GRU = 32
# fully connected dqn
class FC_GRU(nn.Module):
	def __init__(self, input_shape, num_actions):
		super(FC_GRU, self).__init__()

		# first fully connected layers
		self.fc = nn.Sequential(nn.Linear(input_shape, HIDDEN_SHAPE_FIRST),
								nn.ReLU(),
								nn.Linear(HIDDEN_SHAPE_FIRST, HIDDEN_SHAPE_SECOND))
		self.gru = nn.GRU(input_size=HIDDEN_SHAPE_SECOND, hidden_size=HIDDEN_SHAPE_GRU)
		self.fc2 =nn.Sequential(nn.Linear(HIDDEN_SHAPE_GRU, num_actions))

	def forward(self, x:PackedSequence):
		fc_data = self.fc(x.data)
		x = PackedSequence(fc_data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
		_, h_n = self.gru(x) # feed through gru with initial hidden state = zeros
		output = h_n.view(h_n.shape[1], -1)
		output = self.fc2(output)
		return output
	
	def forward_hidden(self, x, hidden):
		fc_data =  self.fc(x)
		fc_data = fc_data.view(1, 1, -1)
		_, h_n = self.gru(fc_data, hidden)
		output = h_n.view(h_n.shape[1], -1)
		output = self.fc2(output)
		return output, h_n

	def getInitialHidden(self):
		h = torch.zeros(1, 1, HIDDEN_SHAPE_GRU)
		return h
