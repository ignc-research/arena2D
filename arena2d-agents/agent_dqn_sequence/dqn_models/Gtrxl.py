from dqn_models import transformer_xl as txl
from torch.nn.utils.rnn import PackedSequence
import copy
import torch.nn as nn
import torch
import math
from torch.autograd import Variable

### hyper parameters ###
n_head = 6	                # number of heads for multi-head self attention
d_ff = 256	                # number of neurons in feedforward layer of transformer
embedding_size=384              # number of output neurons of the embedding layer
dropout = 0.1	                # dropout applied after each transformer submodule
dropatt= 0.1
men_len= 36
##################################

class TransformerDqn(nn.Module):
	def __init__(self, input_size, output_size,n_layer):
		super(TransformerDqn, self).__init__()
		
		self.embedding = nn.Linear(input_size,embedding_size)

		# create transformer with N layers											
		self.transformer = txl.MemTransformerLM(n_layer,n_head,embedding_size,embedding_size//n_head, d_ff,dropout, dropatt)

		# map output of final layer to Q values of actions
		self.linear = nn.Linear(embedding_size, output_size)

	# x is a list of N sequences or a tensor (sequence, batch, observation)
	# returns a tensor (N, NUM_ACTIONS) containing q values for 
	def forward(self, x, mem):
		# sequences with different lengths
                embedding=self.embedding(x.data)                
                x_transform, new_mem = self.transformer(embedding, mem)
                q_vals = self.linear(x_transform[-1])
                return q_vals, new_mem
                
