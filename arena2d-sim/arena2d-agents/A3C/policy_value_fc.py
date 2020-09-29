import torch
import torch.nn as nn

HIDDEN_SHAPE_BODY=64
HIDDEN_SHAPE_BODY_OUT=64
HIDDEN_SHAPE_POLICY=32
HIDDEN_SHAPE_VALUE=32

class PolicyValueFC(nn.Module):
	def __init__(self, num_inputs, num_actions):
		super(PolicyValueFC, self).__init__()
		self.body = nn.Sequential(	nn.Linear(num_inputs, HIDDEN_SHAPE_BODY),
									nn.ReLU(),
									nn.Linear(HIDDEN_SHAPE_BODY, HIDDEN_SHAPE_BODY_OUT))
		self.policy = nn.Sequential(nn.Linear(HIDDEN_SHAPE_BODY, HIDDEN_SHAPE_POLICY),
									nn.ReLU(),
									nn.Linear(HIDDEN_SHAPE_POLICY, num_actions));	

		self.value = nn.Sequential(	nn.Linear(HIDDEN_SHAPE_BODY, HIDDEN_SHAPE_VALUE),
									nn.ReLU(),
									nn.Linear(HIDDEN_SHAPE_VALUE, 1))
								
	def forward(self, x):
		body_x = self.body(x)
		policy_x = self.policy(body_x)
		value_x = self.value(body_x)
		return policy_x, value_x
	
	def get_value(self, x):
		body_x = self.body(x)
		return self.value(body_x)
	
	def get_policy(self, x):
		body_x = self.body(x)
		return self.policy(body_x)
