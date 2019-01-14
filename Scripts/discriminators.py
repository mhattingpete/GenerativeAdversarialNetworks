# global imports
from torch import nn
import torch

# local imports
from layers import MultiLayerPerceptron,MiniBatchStd,Conv2dEqualized,LinearEqualized,SelfAttention,MemoryCell

#######################################
#####    Unconditional models     #####
#######################################

class Discriminator(nn.Module):
	def __init__(self,hidden_sizes,dropout_prob=0.1,activation=nn.ELU(),last_activation=nn.Sigmoid()):
		super().__init__()
		self.layers = MultiLayerPerceptron(hidden_sizes,activation=activation,dropout_prob=dropout_prob,last_activation=last_activation)

	def forward(self,x):
		return self.layers(x)

class ConvDiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,activation=nn.LeakyReLU(0.2),last_activation=None):
		super().__init__()
		# layers
		layers = [
		nn.Conv2d(input_size,hidden_size,kernel_size=4,stride=2,padding=1),
		activation,
		nn.Conv2d(hidden_size,hidden_size*2,kernel_size=4,stride=2,padding=1),
		activation,
		nn.Conv2d(hidden_size*2,hidden_size*4,kernel_size=4,stride=2,padding=0),
		activation,
		nn.Conv2d(hidden_size*4,output_size,kernel_size=2,stride=1,padding=0)
		]
		if last_activation:
			layers.append(last_activation)
		self.layers = nn.Sequential(*layers)
		self.output_size = output_size

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				m.weight.data.normal_(0.0,0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self,x):
		return self.layers(x).view(-1,self.output_size)

class SADiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,activation=nn.LeakyReLU(0.2),last_activation=None):
		super().__init__()
		# layers
		layers = [
		nn.utils.spectral_norm(nn.Conv2d(input_size,hidden_size,kernel_size=4,stride=2,padding=1)),
		activation,
		nn.utils.spectral_norm(nn.Conv2d(hidden_size,hidden_size*2,kernel_size=4,stride=2,padding=1)),
		activation,
		SelfAttention(hidden_size*2),
		activation,
		nn.utils.spectral_norm(nn.Conv2d(hidden_size*2,hidden_size*4,kernel_size=4,stride=2,padding=0)),
		activation,
		nn.utils.spectral_norm(nn.Conv2d(hidden_size*4,output_size,kernel_size=2,stride=1,padding=0))
		]
		if last_activation:
			layers.append(last_activation)
		self.layers = nn.Sequential(*layers)
		self.output_size = output_size

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				m.weight.data.normal_(0.0,0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self,x):
		return self.layers(x).view(-1,self.output_size)

class PGDiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		# internal variables
		self.activation = activation
		self.input_size = input_size
		self.block_output_size = hidden_size
		# layers
		block = nn.Sequential(
			MiniBatchStd(),
			Conv2dEqualized(hidden_size+1,hidden_size,kernel_size=3,stride=1,padding=1),
			self.activation,
			Conv2dEqualized(hidden_size,hidden_size,kernel_size=4,stride=1,padding=0),
			self.activation
		)
		self.fromRGB = Conv2dEqualized(input_size,hidden_size,kernel_size=1,stride=1,padding=0)
		self.layers = nn.ModuleList([block])
		self.toOut = LinearEqualized(hidden_size,1)
		self.avg_pool = nn.AvgPool2d(kernel_size=2)
		self.block_ready = 0

	def create_next_block(self,hidden_size):
		self.block_ready = 1
		self.new_fromRGB = Conv2dEqualized(self.input_size,hidden_size,kernel_size=1,stride=1,padding=0)
		self.next_block = nn.Sequential(
			Conv2dEqualized(hidden_size,hidden_size,kernel_size=3,stride=1,padding=1),
			self.activation,
			Conv2dEqualized(hidden_size,self.block_output_size,kernel_size=3,stride=1,padding=1),
			self.activation,
			self.avg_pool
		)
		self.block_output_size = hidden_size

	def add_next_block(self):
		# add next block of layers to model
		self.block_ready = 0
		tmp = self.layers
		self.layers = nn.ModuleList([self.next_block])
		self.layers.extend(tmp)
		self.fromRGB = self.new_fromRGB

	def fade_in_layer(self,x,alpha):
		x_new = self.new_fromRGB(x)
		x_new = self.next_block(x_new)
		x = self.avg_pool(x)
		x = self.fromRGB(x)
		x = torch.add(x.mul(1.0-alpha),x_new.mul(alpha))
		for l in self.layers:
			x = l(x)
		return self.toOut(x.view(x.size(0),-1))

	def forward(self,x,alpha=1.0):
		if alpha < 1.0:
			return self.fade_in_layer(x,alpha)
		else:
			if self.block_ready:
				self.add_next_block()
			x = self.fromRGB(x)
			for l in self.layers:
				x = l(x)
			return self.toOut(x.view(x.size(0),-1))

#######################################
#####     Conditional models      #####
#######################################

class CondConvDiscriminator(nn.Module):
	def __init__(self,input_size,conditional_size,hidden_size,output_size,activation=nn.LeakyReLU(0.2),last_activation=None):
		super().__init__()
		# conditional layers
		cond_layers = [
		nn.Conv2d(conditional_size,hidden_size,kernel_size=4,stride=2,padding=1),
		activation
		]
		self.cond_layers = nn.Sequential(*cond_layers)
		# input layers
		input_layers = [
		nn.Conv2d(input_size,hidden_size,kernel_size=4,stride=2,padding=1),
		activation
		]
		self.input_layers = nn.Sequential(*input_layers)

		# concatenated input layers
		layers = [
		nn.Conv2d(hidden_size*2,hidden_size*2,kernel_size=4,stride=2,padding=1),
		activation,
		nn.Conv2d(hidden_size*2,hidden_size*4,kernel_size=4,stride=2,padding=0),
		activation,
		nn.Conv2d(hidden_size*4,output_size,kernel_size=2,stride=1,padding=0)
		]
		if last_activation:
			layers.append(last_activation)
		self.layers = nn.Sequential(*layers)
		self.output_size = output_size

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				m.weight.data.normal_(0.0,0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self,x,conditional):
		y = self.cond_layers(conditional)
		x = self.input_layers(x)
		x = torch.cat([x,y],1)
		return self.layers(x).view(-1,self.output_size)

#######################################
#####        Text models          #####
#######################################

class GumbelRNNDiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		# layers
		self.output_size = output_size
		self.embedding = nn.Linear(input_size,hidden_size)
		self.batchnorm = nn.BatchNorm1d(hidden_size)
		self.activation = activation
		self.rnn = nn.GRU(hidden_size,output_size,batch_first=True)

	def forward(self,x):
		x = self.batchnorm(self.activation(self.embedding(x)).transpose(2,1)).transpose(1,2)
		_,x = self.rnn(x)
		return x.view(-1,self.output_size)

class GumbelSARNNDiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		# layers
		self.output_size = output_size
		self.embedding = nn.Linear(input_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(hidden_size)
		self.attention = SelfAttention(hidden_size,layer_type="conv1d")
		self.batchnorm2 = nn.BatchNorm1d(hidden_size)
		self.activation = activation
		self.rnn = nn.GRU(hidden_size,output_size,batch_first=True)

	def forward(self,x):
		x = self.batchnorm1(self.activation(self.embedding(x)).transpose(2,1))
		x = self.batchnorm2(self.activation(self.attention(x))).transpose(1,2)
		_,x = self.rnn(x)
		return x.view(-1,self.output_size)

class GumbelRelRNNDiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,rnn_size,output_size,activation=nn.LeakyReLU(0.2),num_embeddings=6):
		super().__init__()
		# layers
		self.num_embeddings = num_embeddings
		self.output_size = output_size
		self.embeddings = nn.ModuleList([nn.Sequential(nn.Linear(input_size,hidden_size),activation) for _ in range(num_embeddings)])
		self.attention = SelfAttention(hidden_size,layer_type="conv1d")
		self.batchnorm = nn.BatchNorm1d(hidden_size)
		self.linear = nn.Linear(hidden_size,rnn_size)
		self.activation = activation
		self.rnn = nn.GRU(rnn_size,output_size,batch_first=True)

	def forward(self,x):
		x_embedded = []
		for l in self.embeddings:
			x_emb = l(x).transpose(2,1)
			x_emb = self.batchnorm(self.activation(self.attention(x_emb))).transpose(1,2)
			x_emb = self.activation(self.linear(x_emb))
			_,x_emb = self.rnn(x_emb)
			x_embedded.append(x_emb)
		x = torch.stack(x_embedded,dim=1)
		return x.view(-1,self.num_embeddings,self.output_size)

class MemoryDiscriminator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,max_seq_len,activation=nn.LeakyReLU(0.2),sim_size=4,similarity=nn.CosineSimilarity(dim=-1)):
		super().__init__()
		self.output_size = output_size
		self.embedding = nn.Linear(input_size,hidden_size)
		self.batchnorm = nn.BatchNorm1d(hidden_size)
		self.activation = activation
		self.memcell = MemoryCell(hidden_size,output_size,sim_size=sim_size,similarity=similarity,use_fused=False)
		self.max_seq_len = max_seq_len
		self.memory = nn.Parameter(torch.randn(1,self.max_seq_len,hidden_size))

	def forward(self,x):
		x = self.batchnorm(self.activation(self.embedding(x)).transpose(2,1)).transpose(1,2)
		memory = self.memory
		memory = memory[:x.size(1),:]
		memory = memory.expand(x.size(0),-1,-1) # copy the memory for each batch position
		hx = None
		hm = None
		for i in range(x.size(1)):
			out,hx,hm = self.memcell(x[:,i,:],memory[:,i,:],hx=hx,hm=hm)
		return out.view(-1,self.output_size)