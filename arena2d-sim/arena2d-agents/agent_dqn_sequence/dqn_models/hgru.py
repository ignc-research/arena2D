import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

hidden_dim = 64
HIDDEN_SHAPE = 32              # number of nodes in hidden layer of the fc network
layer_dim = 3                  # number of gru layers


class GRUModel(nn.Module):
    def __init__(self, input_dim,n_actions, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim) #GRUCell(input_dim, hidden_dim, layer_dim) , dropout=0.1
        self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, n_actions))    #nn.Linear(hidden_dim, HIDDEN_SHAPE), nn.ReLU(),
	    
    def forward(self, x:PackedSequence, h):
        out, h = self.gru(x, h)
        out = self.fc(h[-1,:,:])        
        return out,h.data

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.layer_dim, batch_size, self.hidden_dim).zero_().cuda()
        return hidden
