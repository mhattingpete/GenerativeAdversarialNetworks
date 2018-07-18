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