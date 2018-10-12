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
	def __init__(self,hidden_size,layer_type="Conv2d"):
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
		return self.softmax(gumbel_sample/temperature)

	def __repr__(self):
		return self.__class__.__name__ +"()"
