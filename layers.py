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
	def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True):
		super().__init__()
		self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
		nn.init.kaiming_normal_(self.conv.weight,a=nn.init.calculate_gain('conv2d'))
		self.scale = nn.Parameter((torch.mean(self.conv.weight.data**2))**0.5).type(torch.FloatTensor)
		self.conv.weight.data.copy_(self.conv.weight.data/self.scale)
		self.bias = nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

	def forward(self,x):
		x = self.conv(x.mul(self.scale))
		return x+self.bias.view(1,-1,1,1).expand_as(x)

class LinearEqualized(nn.Module):
	def __init__(self,in_features,out_features,bias=True):
		super().__init__()
		self.linear = nn.Linear(in_features,out_features,bias=bias)
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
