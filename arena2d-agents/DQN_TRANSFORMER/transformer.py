import torch
import torch.nn as nn
import copy

# Base Transformer (TrXL) encoder layer from paper "Attention is all you need"
class TransformerLayer(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
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
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
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
class TransformerGLayer(TransformerLayer):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
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
	def __init__(self, d_model, nhead, num_layers=6, transformer_type="base", dim_feedforward=2048, dropout=0.1):
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

def __init__(self, d_model, dropout=0.1, max_len=5000):
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
