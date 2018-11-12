# global imports
from torch import nn
import torch

# local imports
from layers import MultiLayerPerceptron,PixelwiseNormalization,Conv2dEqualized,SelfAttention,GumbelSoftmax,RelationalRNNCell

#######################################
#####    Unconditional models     #####
#######################################

class Generator(nn.Module):
	def __init__(self,hidden_sizes,dropout_prob=0.1,activation=nn.ELU(),last_activation=nn.Tanh()):
		super().__init__()
		self.layers = MultiLayerPerceptron(hidden_sizes,activation=activation,dropout_prob=dropout_prob,last_activation=last_activation)

	def forward(self,x):
		return self.layers(x)

class ConvGenerator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,activation=nn.ReLU(),last_activation=nn.Tanh()):
		super().__init__()
		# layers
		layers = [
		nn.ConvTranspose2d(input_size,hidden_size*4,kernel_size=4,stride=1,padding=0),
		nn.BatchNorm2d(hidden_size*4),
		activation,
		nn.ConvTranspose2d(hidden_size*4,hidden_size*2,kernel_size=4,stride=2,padding=1),
		nn.BatchNorm2d(hidden_size*2),
		activation,
		nn.ConvTranspose2d(hidden_size*2,hidden_size,kernel_size=4,stride=2,padding=1),
		nn.BatchNorm2d(hidden_size),
		activation,
		nn.ConvTranspose2d(hidden_size,output_size,kernel_size=4,stride=2,padding=3),
		last_activation
		]
		self.layers = nn.Sequential(*layers)

		for m in self.modules():
			if isinstance(m,nn.ConvTranspose2d):
				m.weight.data.normal_(0.0,0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self,x):
		return self.layers(x)

class SAGenerator(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,activation=nn.ReLU(),last_activation=nn.Tanh()):
		super().__init__()
		# layers
		layers = [
		nn.utils.spectral_norm(nn.ConvTranspose2d(input_size,hidden_size*4,kernel_size=4,stride=1,padding=0)),
		nn.BatchNorm2d(hidden_size*4),
		activation,
		nn.utils.spectral_norm(nn.ConvTranspose2d(hidden_size*4,hidden_size*2,kernel_size=4,stride=2,padding=1)),
		nn.BatchNorm2d(hidden_size*2),
		activation,
		SelfAttention(hidden_size*2),
		nn.BatchNorm2d(hidden_size*2),
		activation,
		nn.utils.spectral_norm(nn.ConvTranspose2d(hidden_size*2,hidden_size,kernel_size=4,stride=2,padding=1)),
		nn.BatchNorm2d(hidden_size),
		activation,
		nn.utils.spectral_norm(nn.ConvTranspose2d(hidden_size,output_size,kernel_size=4,stride=2,padding=3)),
		last_activation
		]
		self.layers = nn.Sequential(*layers)

		for m in self.modules():
			if isinstance(m,nn.ConvTranspose2d):
				m.weight.data.normal_(0.0,0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self,x):
		return self.layers(x)

class PGGenerator(nn.Module):
	def __init__(self,input_size,output_size,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		# internal variables
		self.activation = activation
		self.input_size = input_size
		self.output_size = output_size
		# layers
		block = nn.Sequential(
			Conv2dEqualized(input_size,input_size,kernel_size=4,stride=1,padding=3),
			self.activation,
			PixelwiseNormalization(),
			Conv2dEqualized(input_size,input_size,kernel_size=3,stride=1,padding=1),
			self.activation,
			PixelwiseNormalization()
		)
		self.layers = nn.ModuleList([block])
		self.toRGB = Conv2dEqualized(input_size,output_size,kernel_size=1,stride=1,padding=0)
		self.block_ready = 0
	
	def create_next_block(self,hidden_size):
		self.block_ready = 1
		self.next_block = nn.Sequential(
			Conv2dEqualized(self.input_size,hidden_size,kernel_size=3,stride=1,padding=1),
			self.activation,
			PixelwiseNormalization(),
			Conv2dEqualized(hidden_size,hidden_size,kernel_size=3,stride=1,padding=1),
			self.activation,
			PixelwiseNormalization()
		)
		self.layers.append(nn.Upsample(scale_factor=2,mode="nearest"))
		self.new_toRGB = Conv2dEqualized(hidden_size,self.output_size,kernel_size=1,stride=1,padding=0)
		self.input_size = hidden_size

	def add_next_block(self):
		# add next block of layers to model
		self.block_ready = 0
		self.layers.append(self.next_block)
		self.toRGB = self.new_toRGB

	def fade_in_layer(self,x,alpha):
		for l in self.layers:
			x = l(x)
		x_new = self.next_block(x)
		x = self.toRGB(x)
		x_new = self.new_toRGB(x_new)
		return torch.add(x.mul(1.0-alpha),x_new.mul(alpha))

	def forward(self,x,alpha=1.0):
		if alpha < 1.0:
			return self.fade_in_layer(x,alpha)
		else:
			if self.block_ready:
				self.add_next_block()
			for l in self.layers:
				x = l(x)
			return self.toRGB(x)

#######################################
#####     Conditional models      #####
#######################################

class CondConvGenerator(nn.Module):
	def __init__(self,input_size,conditional_size,hidden_size,output_size,activation=nn.ReLU(),last_activation=nn.Tanh()):
		super().__init__()
		# conditional layers
		cond_layers = [
		nn.ConvTranspose2d(conditional_size,hidden_size*2,kernel_size=4,stride=1,padding=0),
		nn.BatchNorm2d(hidden_size*2),
		activation
		]
		self.cond_layers = nn.Sequential(*cond_layers)

		# input layers
		input_layers = [
		nn.ConvTranspose2d(input_size,hidden_size*2,kernel_size=4,stride=1,padding=0),
		nn.BatchNorm2d(hidden_size*2),
		activation
		]
		self.input_layers = nn.Sequential(*input_layers)

		# remaining of layers for concatenated input
		layers = [
		nn.ConvTranspose2d(hidden_size*4,hidden_size*2,kernel_size=4,stride=2,padding=1),
		nn.BatchNorm2d(hidden_size*2),
		activation,
		nn.ConvTranspose2d(hidden_size*2,hidden_size,kernel_size=4,stride=2,padding=1),
		nn.BatchNorm2d(hidden_size),
		activation,
		nn.ConvTranspose2d(hidden_size,output_size,kernel_size=4,stride=2,padding=3),
		last_activation
		]
		self.layers = nn.Sequential(*layers)

		for m in self.modules():
			if isinstance(m,nn.ConvTranspose2d):
				m.weight.data.normal_(0.0,0.02)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self,x,conditional):
		y = self.cond_layers(conditional)
		x = self.input_layers(x)
		x = torch.cat([x,y],1)
		return self.layers(x)

#######################################
#####        Text models          #####
#######################################

class GumbelRNNGenerator(nn.Module):
	def __init__(self,hidden_size,noise_size,output_size,device,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		self.device = device
		# internal variable sizes
		self.hidden_size = hidden_size
		step_input_size = hidden_size + hidden_size
		# layer definitions
		self.z2h = nn.Linear(noise_size,hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(step_input_size)
		self.gru = nn.GRUCell(step_input_size,hidden_size)
		self.h2o = nn.Linear(hidden_size,output_size)
		self.batchnorm2 = nn.BatchNorm1d(hidden_size)
		self.activation = activation
		self.last_activation = GumbelSoftmax(device)
		self.EOS_TOKEN = output_size-1

	def forward(self,z,num_steps,temperature,x=None):
		predictions = []
		z = self.activation(self.z2h(z))
		h = z # initialize the hidden state
		previous_output = torch.zeros(z.size(0),dtype=torch.long).to(self.device)
		previous_output[:] = self.EOS_TOKEN # <EOS> token
		for i in range(num_steps):
			previous_output = self.activation(self.embedding(previous_output))
			step_input = torch.cat([previous_output,z],dim=1)
			step_input = self.batchnorm1(step_input)
			h = self.gru(step_input,h)
			out = self.activation(h)
			out = self.batchnorm2(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out,dim=-1)
			predictions.append(out)
		output = torch.stack(predictions).transpose(1,0)
		return output

class GumbelSARNNGenerator(nn.Module):
	def __init__(self,hidden_size,noise_size,output_size,device,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		self.device = device
		# internal variable sizes
		self.hidden_size = hidden_size
		step_input_size = hidden_size + hidden_size
		# layer definitions
		self.z2h = nn.Linear(noise_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(hidden_size)
		self.attention = SelfAttention(hidden_size,layer_type="linear")
		self.batchnorm2 = nn.BatchNorm1d(hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.batchnorm3 = nn.BatchNorm1d(step_input_size)
		self.gru = nn.GRUCell(step_input_size,hidden_size)
		self.h2o = nn.Linear(hidden_size,output_size)
		self.batchnorm4 = nn.BatchNorm1d(hidden_size)
		self.activation = activation
		self.last_activation = GumbelSoftmax(device)
		self.EOS_TOKEN = output_size-1

	def forward(self,z,num_steps,temperature,x=None):
		predictions = []
		z = self.batchnorm1(self.activation(self.z2h(z)))
		z = self.batchnorm2(self.activation(self.attention(z)))
		h = z # initialize the hidden state
		previous_output = torch.zeros(z.size(0),dtype=torch.long).to(self.device)
		previous_output[:] = self.EOS_TOKEN # <EOS> token
		for i in range(num_steps):
			previous_output = self.activation(self.embedding(previous_output))
			step_input = torch.cat([previous_output,z],dim=1)
			step_input = self.batchnorm3(step_input)
			h = self.gru(step_input,h)
			out = self.activation(h)
			out = self.batchnorm4(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out,dim=-1)
			predictions.append(out)
		output = torch.stack(predictions).transpose(1,0)
		return output

class GumbelRelRNNGenerator(nn.Module):
	def __init__(self,mem_slots,head_size,num_heads,noise_size,output_size,device,activation=nn.LeakyReLU(0.2)):
		super().__init__()
		self.device = device
		# internal variable sizes
		hidden_size = head_size * num_heads
		self.hidden_size = hidden_size
		step_input_size = hidden_size + hidden_size
		self.mem_slots = mem_slots
		# layer definitions
		self.z2m = nn.Linear(noise_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.batchnorm2 = nn.BatchNorm1d(step_input_size)
		self.relRNN = RelationalRNNCell(step_input_size,mem_slots=mem_slots,head_size=head_size,num_heads=num_heads,gate_type=None,activation=activation)
		self.m2o = nn.Linear(mem_slots*hidden_size,output_size)
		self.activation = activation
		self.last_activation = GumbelSoftmax(device)
		self.EOS_TOKEN = output_size-1
		self.memory = None

	def forward(self,z,num_steps,temperature,x=None):
		batch_size = z.size(0)
		if self.memory is None:
			self.initMemory(batch_size)
		predictions = []
		z = self.batchnorm1(self.activation(self.z2m(z)))
		# detach memory such that we don't backprop through the whole dataset
		self.memory = self.memory.detach()
		previous_output = torch.zeros(batch_size,dtype=torch.long).to(self.device)
		previous_output[:] = self.EOS_TOKEN # <EOS> token
		for i in range(num_steps):
			previous_output = self.activation(self.embedding(previous_output))
			step_input = torch.cat([previous_output,z],dim=1)
			step_input = self.batchnorm2(step_input)
			self.memory = self.relRNN(step_input,self.memory)
			out = self.m2o(self.memory.view(self.memory.size(0),-1))
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out,dim=-1)
			predictions.append(out)
		output = torch.stack(predictions).transpose(1,0)
		return output

	def initMemory(self,batch_size):
		self.memory = self.relRNN.initMemory(batch_size)
