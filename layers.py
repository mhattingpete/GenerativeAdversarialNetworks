from torch import nn

class MultiLayerPerceptron(nn.Module):
	def __init__(self,hidden_sizes,activation=nn.ELU(),dropout_prob=0.1,last_activation=nn.ELU()):
		super().__init__()
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