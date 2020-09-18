from dqn_models import vanillaTransformer as vtxl
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
'''

TO DO:
Add in layer normalization to transformer encoder
Remember at test time since we already have a well trained model, I think we don't send in 
extra states anymore when computing the Q-values (can try both). If do send extras then need to make 
sure the dropout is turned off in transformer encoder but not in the first embedding layers. 

Right now dropout scales the outputs during training, is this what we want for those first layers?
Check that dropout implementation in pytorch has different dropout per element in the batch #$$$$$$$$$$


DQN now takes as input a set of states, then feeds each one through the embedder B times,
which will help encoder uncertainty of the embeddings, and then feed the combined results through 
transformer encoder
'''
#embedder_params={'dropout':0.1,'B':3,}
#encoder_layer_params={'d_model':256,'nhead':8}
class CartPoleEmbedder(nn.Module):
    def __init__(self,input_size,dropout=0.1, B=1, embedding_size=384):
        '''
        :param B: Number of times we embed each state (with dropout each time)

        '''

        super(CartPoleEmbedder, self).__init__()
        self.B = B
        self.dropout_p = dropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.layer3 = nn.Linear(embedding_size,embedding_size)
        self.layer4 = nn.Linear(input_size,embedding_size)

        #now need to combine the B copies of the elements
        #Can start by using just linear combo then move to nonlinear combo


        '''
        self.layer3 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU()
        )
        '''

    def forward(self,input,is_training=True):
        #want to now stack B copies of input on top of eachother
        #Batch dim is dim 0
        input = torch.cat(self.B*[input])

        #The dropout implementation in pytorch applies dropout differently per element in batch
        #which is what we want

        #hidden = self.layer1(input)
        #hidden = self.layer2(hidden)
        return self.layer4(input)



class TransformerDqn(nn.Module):

    def __init__(self,output_size,input_size,num_encoder_layers=1):
        '''
        :param embedder: module to embed the states
        :param output_size: number of actions we can choose
        '''

        #dropout = 0.1
        hidden_size=384

        super(TransformerDqn, self).__init__()
        self.embedder = CartPoleEmbedder(input_size=input_size)
        self.pos_encoder = PositionalEncoding(d_model=384, dropout=0.1)
        self.encoder_layer = vtxl.StableTransformerLayer(d_model=384,nhead=6,dim_feedforward=256, dropout=0.1, use_gate = True)
        self.encoder = vtxl.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(hidden_size,output_size)

    def forward(self,input):
        '''
        :param input: matrix of state vectors (last column will contain state of interest)
        :return: vector of Q values for each action
        '''       

        embedding = self.embedder(input)
        embedding = self.pos_encoder(embedding)       
        embedding = self.encoder(embedding)
        #print('embedding.size:',embedding.shape)
        #print('output.size:',self.output_layer(embedding).shape)
        return self.output_layer(embedding).squeeze(1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).double() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe.transpose(0, 1)[:x.size(0), :], requires_grad=False)
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    # Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    # Unmasked positions are filled with float(0.0).
    mask = (torch.triu(torch.ones(sz, sz)) == 1).float().transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask




