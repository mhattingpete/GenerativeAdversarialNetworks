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
	def __init__(self,input_size,mem_slots,head_size,num_heads=8,gate_type=None,num_attention_blocks=3,activation=nn.ELU()):
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
		# memory settings
		self.mem_slots = mem_slots
		self.mem_slots_plus_input = mem_slots + 1 # denoted as N
		self.num_gates = 2*self.gate_size()
		self.mem_size = head_size * num_heads

		# gate settings
		assert gate_type in ["memory","unit",None]
		self.gate_type = gate_type

		# attention settings
		self.num_attention_blocks = num_attention_blocks

		# input projection
		self.i2m = nn.Linear(input_size,self.mem_size) # from input to mem_size
		self.activation = activation

		# attention module
		self.head_size = head_size
		self.qkv_size = 3 * head_size
		self.total_qkv_size = self.qkv_size * num_heads # denoted as F
		self.num_heads = num_heads
		self.qkv_layer = nn.Linear(self.mem_size,self.total_qkv_size,bias=False)
		self.qkv_layernorm = nn.LayerNorm([self.mem_slots_plus_input,self.total_qkv_size])
		self.softmax = nn.Softmax(dim=-1)
		self.att_layernorm1 = nn.LayerNorm([self.mem_slots,self.mem_size])
		self.att_layernorm2 = nn.LayerNorm([self.mem_slots,self.mem_size])
		self.mlp_layers = MultiLayerPerceptron(hidden_sizes=[self.mem_size,self.mem_size,self.mem_size]) # multi layer perceptron module

		# gates
		if self.num_gates > 0:
			self.i2g = nn.Linear(self.mem_size,self.num_gates) # from input (with mem_size) to gate
			self.m2g = nn.Linear(self.mem_size,self.num_gates) # form memory to gate
		else:
			self.i2g = None
			self.m2g = None
		# biases for gates
		self.forget_bias = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
		self.input_bias = nn.Parameter(torch.tensor(0.0,dtype=torch.float32))

		# output size
		self._output_size = self.mem_size * self.mem_slots

	def forward(self,x,memory=None):
		if memory is None:
			raise ValueError("Memory is not initialized please do first by calling initMemory(batch_size)")
		x = self.activation(self.i2m(x)).unsqueeze(1)
		memory_plus_input = torch.cat([memory,x],dim=1)
		next_memory = self.attend_over_memory(memory_plus_input)
		# cut out the memory from concatenated memory and input
		next_memory = next_memory[:,:-x.size(1),:]
		if self.gate_type == "unit" or self.gate_type == "memory":
			next_memory = self.apply_gates(x,memory,next_memory)
		return next_memory

	def attend_over_memory(self,memory):
		for _ in range(self.num_attention_blocks):
			# attend to memory and a skip connection
			memory = self.multihead_attention(memory) + memory
			# add layernorm
			memory = self.att_layernorm1(memory)
			# skip connection to mlp output
			memory = self.mlp_layers(memory) + memory
			# add layernorm
			memory = self.att_layernorm2(memory)
		return memory

	def multihead_attention(self,memory):
		"""
		Perform multi-head attention from the paper 'Attention is All You Need'
		Arxiv url: https://arxiv.org/abs/1706.03762
		
		Args:
		  memory: Memory tensor to perform attention on.
		Returns:
		  next_memory: Next memory tensor.
		"""
		
		qkv = self.qkv_layer(memory)
		qkv = self.qkv_layernorm(qkv)
		
		# split the qkv to multiple heads H
		# [B, N, F] => [B, N, H, F/H]
		qkv = qkv.view(qkv.size(0),self.mem_slots_plus_input,self.num_heads,self.qkv_size)

		# [B, N, H, F/H] => [B, H, N, F/H]
		qkv = qkv.permute(0,2,1,3)

		# split into query, key and value
		# [B, H, N, head_size], [B, H, N, head_size], [B, H, N, head_size]
		q,k,v = torch.split(qkv,[self.head_size,self.head_size,self.head_size],-1)

		# scale q with d_k, the dimensionality of the key vectors
		q *= (self.head_size ** -0.5)

		# make it [B, H, N, N]
		att = torch.matmul(q,k.permute(0,1,3,2))
		att = self.softmax(att)

		# output is [B, H, N, V]
		next_memory = torch.matmul(att,v)

		# [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
		next_memory = next_memory.permute(0,2,1,3)
		next_memory = next_memory.view(next_memory.shape[0],next_memory.shape[1],-1)
		return next_memory


	def apply_gates(self,x,memory,next_memory):
		memory = torch.tanh(memory)
		gate_x = self.i2g(x)
		gate_mem = self.m2g(memory)
		gates = gate_x + gate_mem
		# split gates into input gate and forget gate
		input_gate,forget_gate = torch.split(gates,split_size_or_sections=int(self.num_gates/2),dim=2)
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

	def initMemory(self,batch_size):
		init_memory = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])
		# pad the matrix with zeros
		if self.mem_size > self.mem_slots:
			difference = self.mem_size - self.mem_slots
			pad = torch.zeros((batch_size,self.mem_slots,difference))
			init_memory = torch.cat([init_memory,pad],-1)
		# take the first "self.mem_size" components
		elif self.mem_size < self.mem_slots:
			init_memory = init_memory[:,:,:self.mem_size]
		return init_memory

	@property
	def output_size(self):
		return self._output_size

	def __repr__(self):
		return self.__class__.__name__ +"(memory_size = [?,{},{}])".format(self.mem_slots,self.mem_size)