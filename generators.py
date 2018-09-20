# global imports
from torch import nn
import torch

# local imports
from layers import MultiLayerPerceptron,PixelwiseNormalization,Conv2dEqualized

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