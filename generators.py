# global imports
from torch import nn
import torch

# local imports
from layers import MultiLayerPerceptron

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