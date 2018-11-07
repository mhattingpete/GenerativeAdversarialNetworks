import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
	def __init__(self,hidden_sizes,activation=nn.ELU(),dropout_prob=0.1,last_activation=nn.ELU()):
		super().__init__()
		self.hidden_sizes = hidden_sizes
		# layers
		layers = []
		for i in range(len(hidden_sizes)-1):
			layers.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
			if (i+1 == len(hidden_sizes)-1):
				if last_activation:
					layers.append(last_activation)
			else:
				layers.append(nn.Dropout(dropout_prob))
				layers.append(activation)
		self.mlp_layers = nn.Sequential(*layers)

	def forward(self,x):
		return self.mlp_layers(x)

	def __repr__(self):
		return self.__class__.__name__ +"(hidden_sizes = {})".format(self.hidden_sizes)

class Conv2dEqualized(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1):
		super().__init__()
		self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=False)
		nn.init.kaiming_normal_(self.conv.weight,a=nn.init.calculate_gain('conv2d'))
		self.scale = nn.Parameter((torch.mean(self.conv.weight.data**2))**0.5).type(torch.FloatTensor)
		self.conv.weight.data.copy_(self.conv.weight.data/self.scale)
		self.bias = nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

	def forward(self,x):
		x = self.conv(x.mul(self.scale))
		return x+self.bias.view(1,-1,1,1).expand_as(x)

class LinearEqualized(nn.Module):
	def __init__(self,in_features,out_features):
		super().__init__()
		self.linear = nn.Linear(in_features,out_features,bias=False)
		nn.init.kaiming_normal_(self.linear.weight,a=nn.init.calculate_gain('linear'))
		self.scale = nn.Parameter((torch.mean(self.linear.weight.data**2))**0.5).type(torch.FloatTensor)
		self.linear.weight.data.copy_(self.linear.weight.data/self.scale)
		self.bias = nn.Parameter(torch.FloatTensor(out_features).fill_(0))

	def forward(self,x):
		x = self.linear(x.mul(self.scale))
		return x+self.bias.view(1,-1).expand_as(x)

class MiniBatchStd(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x):
		target_shape = list(x.size())
		target_shape[1] = 1
		std = torch.sqrt(torch.mean((x-torch.mean(x,dim=0,keepdim=True))**2,dim=0,keepdim=True)+1e-8)
		std = torch.mean(std.view(1,-1),dim=1,keepdim=True)
		std = std.expand(*target_shape)
		x = torch.cat((x,std),dim=1)
		return x

	def __repr__(self):
		return self.__class__.__name__ +"()"

class PixelwiseNormalization(nn.Module):
	def __init__(self):
		super().__init__()
		self.eps = 1e-8

	def forward(self,x):
		return x/(torch.mean(x**2,dim=1,keepdim=True)+self.eps)**0.5

	def __repr__(self):
		return self.__class__.__name__ +"(eps = {})".format(self.eps)

class SelfAttention(nn.Module):
	def __init__(self,hidden_size,layer_type="conv2d"):
		super().__init__()
		self.hidden_size = hidden_size
		self.C_bar = int(hidden_size/8)
		layer_type = layer_type.lower()
		if layer_type == "linear":
			self.f = nn.Linear(hidden_size,self.C_bar)
			self.g = nn.Linear(hidden_size,self.C_bar)
			self.h = nn.Linear(hidden_size,hidden_size)
		elif layer_type == "conv1d":
			self.f = nn.Conv1d(hidden_size,self.C_bar,kernel_size=1)
			self.g = nn.Conv1d(hidden_size,self.C_bar,kernel_size=1)
			self.h = nn.Conv1d(hidden_size,hidden_size,kernel_size=1)
		else:
			self.f = nn.Conv2d(hidden_size,self.C_bar,kernel_size=1)
			self.g = nn.Conv2d(hidden_size,self.C_bar,kernel_size=1)
			self.h = nn.Conv2d(hidden_size,hidden_size,kernel_size=1)

		self.gamma = nn.Parameter(torch.FloatTensor(1).fill_(0))
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,x):
		batch_size = x.size(0)
		x_f = self.f(x).view(batch_size,self.C_bar,-1)
		x_g = self.g(x).view(batch_size,self.C_bar,-1)
		x_h = self.h(x).view(batch_size,self.hidden_size,-1)
		s = torch.transpose(x_f,1,2).matmul(x_g)
		beta = self.softmax(s)
		o = x_h.matmul(beta).view(*x.shape)
		return x + o.mul(self.gamma)

	def __repr__(self):
		return self.__class__.__name__ +"(hidden_size = {})".format(self.hidden_size)

class GumbelSoftmax(nn.Module):
	def __init__(self,device):
		super().__init__()
		self.softmax = nn.Softmax(dim=-1)
		self.device = device

	def forward(self,x,temperature):
		eps = 1e-20
		g = -torch.log(-torch.log(torch.rand(*x.shape,device=self.device)+eps)+eps)
		gumbel_sample = x + g
		return self.softmax(gumbel_sample*temperature)

	def __repr__(self):
		return self.__class__.__name__ +"()"

class RelationalRNNCell(nn.Module):
	def __init__(self,input_size,mem_slots,head_size,num_heads=8,gate_type=None,activation=nn.ELU()):
		"""
		input_size: the size of the input at each timestep
		mem_slots: the number of rows of memory
		head_size: the size of the attention head, also used to compute the number of cols in memory
		num_heads: the number of attention heads, head_size x num_heads gives the number of cols in memory
		gate_type: the type of gate to be applied; either 'memory', 'unit' or None, 'memory' specifies that
		 the gates should be applied across the whole memory, 'unit' specifies that the gates should be
		 applied across each unit in the memory and None don't use gates
		activation: the activation applied to the input layer
		"""
		super().__init__()
		assert gate_type in ["memory","unit",None]
		self.gate_type = gate_type
		self.mem_slots = mem_slots
		self.num_gates = 2*self.gate_size()
		self.mem_size = head_size * num_heads
		self.i2m = nn.Linear(input_size,self.mem_size) # from input to mem_size
		self.activation = activation
		self.attention = MultiHeadAttention(self.mem_size,num_heads=num_heads) # multihead attention module
		self.mlp_layers = MultiLayerPerceptron(hidden_sizes=[self.mem_size,2*self.mem_size,self.mem_size]) # multi layer perceptron module
		self.i2g = nn.Linear(self.mem_size,self.num_gates) # from input (with mem_size) to gate
		self.m2g = nn.Linear(self.mem_size,self.num_gates) # form memory to gate
		# biases for gates
		self.forget_bias = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
		self.input_bias = nn.Parameter(torch.tensor(0.0,dtype=torch.float32))
		self._output_size = self.mem_size * self.mem_slots

	def forward(self,x,memory=None):
		if memory is None:
			memory = self.initMemory(x)
		x = self.activation(self.i2m(x)).unsqueeze(1)
		memory_plus_input = torch.cat([memory,x],dim=1)
		pred_memory = self.attention(memory,memory_plus_input) + memory
		next_memory = self.mlp_layers(pred_memory) + pred_memory
		if self.gate_type == 'unit' or self.gate_type == 'memory':
			next_memory = self.apply_gates(x,memory,next_memory)
		return next_memory

	def apply_gates(self,x,memory,next_memory):
		memory = torch.tanh(memory)
		gate_x = self.i2g(x)
		gate_mem = self.m2g(memory)
		gates = gate_x + gate_mem
		input_gate, forget_gate = torch.split(gates,split_size_or_sections=int(self.num_gates/2),dim=2)
		input_gate = torch.sigmoid(input_gate + self.input_bias)
		forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
		next_memory = input_gate * torch.tanh(next_memory)
		next_memory += forget_gate * memory
		return next_memory

	def gate_size(self):
		if self.gate_type == "memory":
			return 1
		elif self.gate_type == "unit":
			return self.mem_size
		else:
			return 0

	def initMemory(self,x):
		eye = x.new_zeros(self.mem_slots,self.mem_size)
		nn.init.eye_(eye)
		return eye.unsqueeze(0).expand(x.size(0),-1,-1)

	@property
	def output_size(self):
		return self._output_size

	def __repr__(self):
		return self.__class__.__name__ +"(memory_size = [?,{},{}])".format(self.mem_slots,self.mem_size)
	

class MultiHeadAttention(nn.Module):
	def __init__(self,hidden_size,num_heads=8):
		super().__init__()
		assert hidden_size % num_heads == 0
		self.hidden_size = hidden_size
		self.hidden_size_tensor = nn.Parameter(torch.tensor(self.hidden_size,dtype=torch.float32),requires_grad=False)
		self.num_heads = num_heads
		self.query_layer = nn.Linear(hidden_size,hidden_size,bias=False)
		self.key_layer = nn.Linear(hidden_size,hidden_size,bias=False)
		self.value_layer = nn.Linear(hidden_size,hidden_size,bias=False)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,query,value):
		Q = self.query_layer(query)
		K = self.key_layer(value)
		V = self.value_layer(value)
		# split Q,K and V into num_heads values from dim=2 and merge in dim=0
		chunk_size = int(self.hidden_size/self.num_heads)
		Q = torch.cat(Q.split(split_size=chunk_size,dim=2),dim=0)
		K = torch.cat(K.split(split_size=chunk_size,dim=2),dim=0)
		V = torch.cat(V.split(split_size=chunk_size,dim=2),dim=0)
		# calculate attention (QK^T)
		att = torch.matmul(Q,K.transpose(1,2))
		# and normalize
		att = att/torch.sqrt(self.hidden_size_tensor)
		# apply softmax
		att = self.softmax(att)
		# multiply with V
		att = torch.matmul(att,V)
		# restore original size
		original_chunk_size = int(att.size(0)/self.num_heads)
		att = torch.cat(att.split(split_size=original_chunk_size,dim=0),dim=2)
		return att
