# global imports
from torch import nn

# local imports
from layers import MultiLayerPerceptron

class Discriminator(nn.Module):
	def __init__(self,hidden_sizes,dropout_prob=0.1,activation=nn.ELU(),last_activation=nn.Sigmoid()):
		super().__init__()
		self.layers = MultiLayerPerceptron(hidden_sizes,activation=activation,dropout_prob=dropout_prob,last_activation=last_activation)

	def forward(self,x):
		return self.layers(x)

class ConvDiscriminator(nn.Module):
	def __init__(self,hidden_sizes,activation=nn.ELU(),last_activation=nn.Sigmoid()):
		super().__init__()
		# layers
		layers = []
		for i in range(len(hidden_sizes)-1):
			layers.append(nn.Conv2d(hidden_sizes[i],hidden_sizes[i+1],kernel_size=(3,3),stride=(1,1)))
			if (i+1 == len(hidden_sizes)-1):
				layers.append(last_activation)
			else:
				layers.append(activation)
		self.layers = nn.Sequential(*layers)

	def forward(self,x):
		return self.layers(x)