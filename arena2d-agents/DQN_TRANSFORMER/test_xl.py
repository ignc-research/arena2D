import transformer_xl
import torch

trafo = transformer_xl.MemTransformerLM(6, 8, 360, 45, 1024, 0.1, 0.1)
t = torch.randn(10, 1, 360)
print("tensor:", t.size())

new_mems = None
h, new_mems = trafo(t, new_mems)
print("hidden:", h.size())
print("new_mems:", len(new_mems)) 
print("new_mems[0]:", new_mems[0].size())
print("new_mems[1]:", new_mems[1].size())
print("new_mems[2]:", new_mems[2].size())
print("new_mems[3]:", new_mems[3].size())
print("new_mems[4]:", new_mems[4].size())
print("new_mems[5]:", new_mems[5].size())
print("new_mems[6]:", new_mems[6].size())
