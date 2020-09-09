import torch
import torch.nn as nn
import copy
import math
import transformer_xl

### transformer hyper parameters ###
TRANSFORMER_TYPE = "id"		# one of "base": standard transformer, "id": identity map reorder, "gated": gated transformer
TRANSFORMER_NUM_LAYERS = 6	# number of encoder layers in transformer
TRANSFORMER_NUM_HEADS = 8	# number of heads for multi-head self attention, make sure num_observations is divisible by this number
TRANSFORMER_FF_DIM = 512	# number of neurons in feedforward layer of transformer
TRANSFORMER_DROPOUT = 0.1	# dropout applied after each transformer submodule
##################################

class QTransformer(nn.Module):
	def __init__(self, num_observations, num_actions, device):
		super(QTransformer, self).__init__()

		# positional encoding
		self.pos_encoder = PositionalEncoding(num_observations)

		# create transformer with N layers
		'''self.transformer = Transformer(		d_model = num_observations,
											nhead = TRANSFORMER_NUM_HEADS,
											num_layers = TRANSFORMER_NUM_LAYERS,
											transformer_type = TRANSFORMER_TYPE,
											dim_feedforward = TRANSFORMER_FF_DIM,
											dropout = TRANSFORMER_DROPOUT)
											'''
		self.transformer = transformer_xl.MemTransformerLM(	TRANSFORMER_NUM_LAYERS,
															TRANSFORMER_NUM_HEADS,
															num_observations,
															num_observations//TRANSFORMER_NUM_HEADS,
															TRANSFORMER_FF_DIM,
															TRANSFORMER_DROPOUT,
															TRANSFORMER_DROPOUT)

		# map output of final layer to Q values of actions
		self.linear = nn.Linear(num_observations, num_actions)
		
		# save device for forwarding
		self.device = device

	# x is a list of N sequences or a tensor (sequence, batch, observation)
	# returns a tensor (N, NUM_ACTIONS) containing q values for 
	def forward(self, x, mem):
		# sequences with different lengths
		x_transform, new_mem = self.transformer(x, mem)

		# feed batch through linear
		q_vals = self.linear(x_transform[-1])
		return q_vals, new_mem
	
# Base Transformer (TrXL) encoder layer from paper "Attention is all you need"
class TransformerLayer(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, dropout):
		super(TransformerLayer, self).__init__()

		# self attention layer
		self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # feed forward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		# normalization layers and dropout
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		# relu activation
		self.activation = nn.ReLU()
	
	def forward(self, src:torch.Tensor):
		# multi head attention and norm
		src2 = self.self_attn(src, src, src)[0]
		src = src + self.dropout1(src2)
		src = self.norm1(src)

		# feed forward and norm
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src


# Transformer with Identity Map Reordering (TrXL-I) encoder layer
class TransformerILayer(TransformerLayer):
	def __init__(self, d_model, nhead, dim_feedforward, dropout):
		super(TransformerILayer, self).__init__(d_model, nhead, dim_feedforward, dropout)
		self.res_activation1 = nn.ReLU()
		self.res_activation2 = nn.ReLU()

	def forward(self, src:torch.Tensor):
		# norm and multi head attention
		src_norm = self.norm1(src)
		src_att = self.dropout1(self.self_attn(src_norm, src_norm, src_norm)[0])
		src = src + self.res_activation1(src_att)

		# feed forward and norm
		src_norm = self.norm2(src)
		src_feed = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src_norm)))))
		src = src + self.res_activation2(src_feed)
		return src

# Gated Transformer (GTrXL) encoder layer from Paper "Stabilizing Transformers for Reinforcement Learning" (2019, Parisotto, Song, et. al.)
class TransformerGLayer(TransformerILayer):
	def __init__(self, d_model, nhead, dim_feedforward, dropout):
		super(TransformerGLayer, self).__init__(d_model, nhead, dim_feedforward, dropout)
		self.res_activation1 = nn.ReLU()
		self.res_activation2 = nn.ReLU()

	def forward(self, src:torch.Tensor):
		# norm and multi head attention
		src_norm = self.norm1(src)
		src_att = self.dropout1(self.self_attn(src_norm, src_norm, src_norm)[0])
		src = src + self.res_activation1(src_att)

		# feed forward and norm
		src_norm = self.norm2(src)
		src_feed = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src_norm)))))
		src = src + self.res_activation2(src_feed)
		return src

# actual transformer instancing multiple encoder layers
# @param transformer_type one of "base": standard transformer, "id": identity map reorder, "gated": gated transformer
class Transformer(nn.Module):
	def __init__(self, d_model, nhead, num_layers, transformer_type, dim_feedforward, dropout):
		super(Transformer, self).__init__()
		enc = None
		# parse transformer_type, create template layer accordingly
		if transformer_type == "base":
			enc = TransformerLayer(d_model, nhead, dim_feedforward, dropout)
		elif transformer_type == "id":
			enc = TransformerILayer(d_model, nhead, dim_feedforward, dropout)
		elif transformer_type == "gated":
			print("WARNING: Gated Transformer Layer (GTrXL) has not been implemented yet!")
			enc = TransformerGLayer(d_model, nhead, dim_feedforward, dropout)

		assert(enc != None)

		# create encoder N instances of encoder type
		self.encoder_layers = nn.ModuleList([copy.deepcopy(enc) for i in range(num_layers)])
	
	def forward(self, x:torch.Tensor):
		for l in self.encoder_layers:
			x = l(x)
		return x


# relative position encoding from pytorch tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=1000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

