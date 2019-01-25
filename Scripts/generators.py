# global imports
from torch import nn
import torch

# local imports
from layers import MultiLayerPerceptron,PixelwiseNormalization,Conv2dEqualized,SelfAttention,GumbelSoftmax,RelationalRNNCell,MemoryCell
from layers import PositionalEmbedding,TransformerDecoder
from utils import create_target_mask

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
	def __init__(self,hidden_size,noise_size,output_size,activation=nn.LeakyReLU(0.2),SOS_TOKEN=None,beam_width=1):
		super().__init__()
		# internal variable sizes
		self.hidden_size = hidden_size
		self.output_size = output_size
		step_input_size = hidden_size + hidden_size
		# layer definitions
		self.z2h = nn.Linear(noise_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.batchnorm2 = nn.BatchNorm1d(step_input_size)
		self.gru = nn.GRUCell(step_input_size,hidden_size)
		self.h2o = nn.Linear(hidden_size,output_size)
		self.batchnorm3 = nn.BatchNorm1d(hidden_size)
		self.activation = activation
		self.last_activation = GumbelSoftmax()
		self.SOS_TOKEN = SOS_TOKEN if SOS_TOKEN is not None else output_size-1
		self.beam_width = beam_width

	def forward(self,z,num_steps,temperature,x=None):
		if self.beam_width > 1 and x is None:
			return self.forward_beam(z=z,num_steps=num_steps,temperature=temperature)
		else:
			return self.forward_greedy(z=z,num_steps=num_steps,temperature=temperature,x=x)

	def forward_beam(self,z,num_steps,temperature):
		predictions = []
		batch_size = z.size(0)
		z = self.batchnorm1(self.activation(self.z2h(z))).repeat(self.beam_width,1)
		h = z # initialize the hidden state
		previous_output = z.new_zeros(size=(batch_size*self.beam_width,),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		# a table for storing the scores
		scores = z.new_zeros(size=(batch_size*self.beam_width,self.output_size))
		# an array of numbers for displacement ie. if batch_size is 2 and beam_width is 3 then this is [0,0,0,3,3,3]. This is used later for indexing
		beam_displacement = torch.arange(start=0,end=batch_size*self.beam_width,step=self.beam_width,dtype=torch.long,device=z.device).view(-1,1).repeat(1,self.beam_width).view(-1)
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			step_input = torch.cat([input,z],dim=1)
			step_input = self.batchnorm2(step_input)
			h = self.gru(step_input,h)
			out = self.activation(h)
			out = self.batchnorm3(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			# compute new scores
			next_scores = scores + torch.log(out+1e-8)
			# select top-k scores where k is the beam width
			score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
			# flatten the output
			outputs = outputs.view(-1)
			# get the indices in the original onehot output by finding the module of the vocab size
			indices = torch.fmod(outputs,self.output_size)
			# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
			beam_indices = torch.div(outputs,self.output_size) + beam_displacement
			# check if some elements/words are repeated
			res = torch.eq(previous_output,indices).nonzero()
			# some elements/words is repeated
			retries = 0
			while res.shape[0] > 0:
				mask = torch.ones(size=(batch_size*self.beam_width,self.output_size),requires_grad=False,device=z.device)
				# set the mask to be zero when an option is non selectable
				mask[beam_indices[res],indices[res]] = 0
				# apply the mask
				out = out * mask
				# set the score for the repeated elements to be low
				next_scores = scores + torch.log(out+1e-8)
				# select top-k scores where k is the beam width
				score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
				# flatten the output
				outputs = outputs.view(-1)
				# get the indices in the original onehot output by finding the module of the vocab size
				indices = torch.fmod(outputs,self.output_size)
				# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
				beam_indices = torch.div(outputs,self.output_size) + beam_displacement
				# check if some elements/words are repeated
				res = torch.eq(previous_output,indices).nonzero()
				if retries > 10:
					break
				retries += 1
			# copy the score for each selected candidate
			scores = score.view(-1,1).repeat(1,self.output_size)
			# renormalize the output
			out = out/out.sum(-1).view(-1,1).repeat(1,self.output_size)
			# append the prediction to output
			predictions.append(out[beam_indices,:])
			# detach the output such that we don't backpropagate through timesteps
			previous_output = indices.detach()
		output = torch.stack(predictions).transpose(1,0)
		# initialize an output_mask such that we can filter out sentences
		output_mask = torch.zeros_like(output)
		# set the selected sentences output_mask to 1
		output_mask[scores[:,0].view(batch_size,-1).argmax(dim=-1) + beam_displacement.view(batch_size,-1)[:,0]] = 1
		# collect the best prediction for each sample in batch
		output = (output*output_mask).view(batch_size,self.beam_width,num_steps,self.output_size)
		# sum the beam sentences. Since the sentences that is not selected is zero this doesn't change the actual sentences
		output = output.sum(1)
		return output

	def forward_greedy(self,z,num_steps,temperature,x=None):
		predictions = []
		z = self.batchnorm1(self.activation(self.z2h(z)))
		h = z # initialize the hidden state
		previous_output = z.new_zeros(size=(z.size(0),),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			step_input = torch.cat([input,z],dim=1)
			step_input = self.batchnorm2(step_input)
			h = self.gru(step_input,h)
			out = self.activation(h)
			out = self.batchnorm3(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out,dim=-1)
				previous_output = previous_output.detach()
			predictions.append(out)
		output = torch.stack(predictions).transpose(1,0)
		return output

class GumbelSARNNGenerator(nn.Module):
	def __init__(self,hidden_size,noise_size,output_size,activation=nn.LeakyReLU(0.2),SOS_TOKEN=None,beam_width=1):
		super().__init__()
		# internal variable sizes
		self.output_size = output_size
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
		self.last_activation = GumbelSoftmax()
		self.SOS_TOKEN = SOS_TOKEN if SOS_TOKEN is not None else output_size-1
		self.beam_width = beam_width

	def forward(self,z,num_steps,temperature,x=None):
		if self.beam_width > 1 and x is None:
			return self.forward_beam(z=z,num_steps=num_steps,temperature=temperature)
		else:
			return self.forward_greedy(z=z,num_steps=num_steps,temperature=temperature,x=x)

	def forward_beam(self,z,num_steps,temperature):
		predictions = []
		batch_size = z.size(0)
		z = self.batchnorm1(self.activation(self.z2h(z)))
		z = self.batchnorm2(self.activation(self.attention(z))).repeat(self.beam_width,1)
		h = z # initialize the hidden state
		previous_output = z.new_zeros(size=(batch_size*self.beam_width,),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		# a table for storing the scores
		scores = z.new_zeros(size=(batch_size*self.beam_width,self.output_size))
		# an array of numbers for displacement ie. if batch_size is 2 and beam_width is 3 then this is [0,0,0,3,3,3]. This is used later for indexing
		beam_displacement = torch.arange(start=0,end=batch_size*self.beam_width,step=self.beam_width,dtype=torch.long,device=z.device).view(-1,1).repeat(1,self.beam_width).view(-1)
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			step_input = torch.cat([input,z],dim=1)
			step_input = self.batchnorm3(step_input)
			h = self.gru(step_input,h)
			out = self.activation(h)
			out = self.batchnorm4(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			# compute new scores
			next_scores = scores + torch.log(out+1e-8)
			# select top-k scores where k is the beam width
			score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
			# flatten the output
			outputs = outputs.view(-1)
			# get the indices in the original onehot output by finding the module of the vocab size
			indices = torch.fmod(outputs,self.output_size)
			# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
			beam_indices = torch.div(outputs,self.output_size) + beam_displacement
			# check if some elements/words are repeated
			res = torch.eq(previous_output,indices).nonzero()
			# some elements/words is repeated
			retries = 0
			while res.shape[0] > 0:
				mask = torch.ones(size=(batch_size*self.beam_width,self.output_size),requires_grad=False,device=z.device)
				# set the mask to be zero when an option is non selectable
				mask[beam_indices[res],indices[res]] = 0
				# apply the mask
				out = out * mask
				# set the score for the repeated elements to be low
				next_scores = scores + torch.log(out+1e-8)
				# select top-k scores where k is the beam width
				score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
				# flatten the output
				outputs = outputs.view(-1)
				# get the indices in the original onehot output by finding the module of the vocab size
				indices = torch.fmod(outputs,self.output_size)
				# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
				beam_indices = torch.div(outputs,self.output_size) + beam_displacement
				# check if some elements/words are repeated
				res = torch.eq(previous_output,indices).nonzero()
				if retries > 10:
					break
				retries += 1
			# copy the score for each selected candidate
			scores = score.view(-1,1).repeat(1,self.output_size)
			# renormalize the output
			out = out/out.sum(-1).view(-1,1).repeat(1,self.output_size)
			# append the prediction to output
			predictions.append(out[beam_indices,:])
			# detach the output such that we don't backpropagate through timesteps
			previous_output = indices.detach()
		output = torch.stack(predictions).transpose(1,0)
		# initialize an output_mask such that we can filter out sentences
		output_mask = torch.zeros_like(output)
		# set the selected sentences output_mask to 1
		output_mask[scores[:,0].view(batch_size,-1).argmax(dim=-1) + beam_displacement.view(batch_size,-1)[:,0]] = 1
		# collect the best prediction for each sample in batch
		output = (output*output_mask).view(batch_size,self.beam_width,num_steps,self.output_size)
		# sum the beam sentences. Since the sentences that is not selected is zero this doesn't change the actual sentences
		output = output.sum(1)
		return output

	def forward_greedy(self,z,num_steps,temperature,x=None):
		predictions = []
		z = self.batchnorm1(self.activation(self.z2h(z)))
		z = self.batchnorm2(self.activation(self.attention(z)))
		h = z # initialize the hidden state
		previous_output = z.new_zeros(size=(z.size(0),),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			step_input = torch.cat([input,z],dim=1)
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
				previous_output = previous_output.detach()
			predictions.append(out)
		output = torch.stack(predictions).transpose(1,0)
		return output

class GumbelRelRNNGenerator(nn.Module):
	def __init__(self,mem_slots,head_size,num_heads,noise_size,output_size,activation=nn.LeakyReLU(0.2),gate_type="memory",dropout_prob=0.2,SOS_TOKEN=None,beam_width=1):
		super().__init__()
		# internal variable sizes
		hidden_size = head_size * num_heads
		self.hidden_size = hidden_size
		step_input_size = hidden_size + hidden_size
		self.mem_slots = mem_slots
		self.output_size = output_size
		# layer definitions
		self.input_dropout = nn.Dropout(p=dropout_prob)
		self.z2m = nn.Linear(noise_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.batchnorm2 = nn.BatchNorm1d(step_input_size)
		self.relRNN = RelationalRNNCell(step_input_size,mem_slots=mem_slots,head_size=head_size,num_heads=num_heads,gate_type=gate_type,activation=activation)
		self.m2o = nn.Linear(mem_slots*hidden_size,output_size)
		self.activation = activation
		self.last_activation = GumbelSoftmax()
		self.SOS_TOKEN = SOS_TOKEN if SOS_TOKEN is not None else output_size-1
		self.beam_width = beam_width

	def forward(self,z,num_steps,temperature,x=None):
		if self.beam_width > 1 and x is None:
			return self.forward_beam(z=z,num_steps=num_steps,temperature=temperature)
		else:
			return self.forward_greedy(z=z,num_steps=num_steps,temperature=temperature,x=x)

	def forward_beam(self,z,num_steps,temperature):
		predictions = []
		batch_size = z.size(0)
		z = self.batchnorm1(self.activation(self.z2m(z))).repeat(self.beam_width,1)
		h = z # initialize the hidden state
		# detach memory such that we don't backprop through the whole dataset
		memory = self.initMemory(batch_size).to(z.device)
		memory = memory.detach()
		previous_output = z.new_zeros(size=(batch_size*self.beam_width,),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		# a table for storing the scores
		scores = z.new_zeros(size=(batch_size*self.beam_width,self.output_size))
		# an array of numbers for displacement ie. if batch_size is 2 and beam_width is 3 then this is [0,0,0,3,3,3]. This is used later for indexing
		beam_displacement = torch.arange(start=0,end=batch_size*self.beam_width,step=self.beam_width,dtype=torch.long,device=z.device).view(-1,1).repeat(1,self.beam_width).view(-1)
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			if x is not None:
				input = self.input_dropout(input)
			step_input = torch.cat([input,z],dim=1)
			step_input = self.batchnorm2(step_input)
			memory = self.relRNN(step_input,memory)
			out = self.m2o(memory.view(memory.size(0),-1))
			out = self.last_activation(out,temperature)
			# compute new scores
			next_scores = scores + torch.log(out+1e-8)
			# select top-k scores where k is the beam width
			score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
			# flatten the output
			outputs = outputs.view(-1)
			# get the indices in the original onehot output by finding the module of the vocab size
			indices = torch.fmod(outputs,self.output_size)
			# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
			beam_indices = torch.div(outputs,self.output_size) + beam_displacement
			# check if some elements/words are repeated
			res = torch.eq(previous_output,indices).nonzero()
			# some elements/words is repeated
			retries = 0
			while res.shape[0] > 0:
				mask = torch.ones(size=(batch_size*self.beam_width,self.output_size),requires_grad=False,device=z.device)
				# set the mask to be zero when an option is non selectable
				mask[beam_indices[res],indices[res]] = 0
				# apply the mask
				out = out * mask
				# set the score for the repeated elements to be low
				next_scores = scores + torch.log(out+1e-8)
				# select top-k scores where k is the beam width
				score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
				# flatten the output
				outputs = outputs.view(-1)
				# get the indices in the original onehot output by finding the module of the vocab size
				indices = torch.fmod(outputs,self.output_size)
				# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
				beam_indices = torch.div(outputs,self.output_size) + beam_displacement
				# check if some elements/words are repeated
				res = torch.eq(previous_output,indices).nonzero()
				if retries > 10:
					break
				retries += 1
			# copy the score for each selected candidate
			scores = score.view(-1,1).repeat(1,self.output_size)
			# renormalize the output
			out = out/out.sum(-1).view(-1,1).repeat(1,self.output_size)
			# append the prediction to output
			predictions.append(out[beam_indices,:])
			# detach the output such that we don't backpropagate through timesteps
			previous_output = indices.detach()
		output = torch.stack(predictions).transpose(1,0)
		# initialize an output_mask such that we can filter out sentences
		output_mask = torch.zeros_like(output)
		# set the selected sentences output_mask to 1
		output_mask[scores[:,0].view(batch_size,-1).argmax(dim=-1) + beam_displacement.view(batch_size,-1)[:,0]] = 1
		# collect the best prediction for each sample in batch
		output = (output*output_mask).view(batch_size,self.beam_width,num_steps,self.output_size)
		# sum the beam sentences. Since the sentences that is not selected is zero this doesn't change the actual sentences
		output = output.sum(1)
		return output

	def forward_greedy(self,z,num_steps,temperature,x=None):
		batch_size = z.size(0)
		predictions = []
		z = self.batchnorm1(self.activation(self.z2m(z)))
		# detach memory such that we don't backprop through the whole dataset
		memory = self.initMemory(batch_size).to(z.device)
		memory = memory.detach()
		previous_output = z.new_zeros(size=(z.size(0),),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			if x is not None:
				input = self.input_dropout(input)
			step_input = torch.cat([previous_output,z],dim=1)
			step_input = self.batchnorm2(step_input)
			memory = self.relRNN(step_input,memory)
			out = self.m2o(memory.view(memory.size(0),-1))
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out,dim=-1)
				previous_output = previous_output.detach()
			predictions.append(out)
		output = torch.stack(predictions).transpose(1,0)
		return output

	def initMemory(self,batch_size):
		return self.relRNN.initMemory(batch_size)

class MemoryGenerator(nn.Module):
	def __init__(self,hidden_size,noise_size,output_size,max_seq_len,activation=nn.LeakyReLU(0.2),sim_size=4,similarity=nn.CosineSimilarity(dim=-1),SOS_TOKEN=None,beam_width=1):
		super().__init__()
		# internal variable sizes
		self.hidden_size = hidden_size
		step_input_size = hidden_size + hidden_size
		self.output_size = output_size
		# layer definitions
		self.z2h = nn.Linear(noise_size,hidden_size)
		self.batchnorm1 = nn.BatchNorm1d(hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.batchnorm2 = nn.BatchNorm1d(step_input_size)
		self.memcell = MemoryCell(step_input_size,hidden_size,sim_size=sim_size,similarity=similarity)
		self.batchnorm3 = nn.BatchNorm1d(hidden_size)
		self.h2o = nn.Linear(hidden_size,output_size)
		self.activation = activation
		self.SOS_TOKEN = SOS_TOKEN if SOS_TOKEN is not None else output_size-1
		self.max_seq_len = max_seq_len
		self.memory = nn.Parameter(torch.randn(1,self.max_seq_len,step_input_size))
		self.last_activation = GumbelSoftmax()
		self.beam_width = beam_width

	def forward(self,z,num_steps,temperature,x=None):
		if self.beam_width > 1 and x is None:
			return self.forward_beam(z=z,num_steps=num_steps,temperature=temperature)
		else:
			return self.forward_greedy(z=z,num_steps=num_steps,temperature=temperature,x=x)

	def forward_beam(self,z,num_steps,temperature):
		if num_steps > self.max_seq_len:
			raise ValueError("num_steps ({}) must be less or equal to max_seq_len ({})".format(num_steps,self.max_seq_len))
		predictions = []
		batch_size = z.size(0)
		z = self.batchnorm1(self.activation(self.z2h(z))).repeat(self.beam_width,1)
		hx = z # initialize the hidden state
		hm = z # initialize the hidden state for memory
		memory = self.memory
		memory = memory[:num_steps,:]
		memory = memory.expand(z.size(0),-1,-1) # copy the memory for each batch position
		previous_output = z.new_zeros(size=(batch_size*self.beam_width,),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		# a table for storing the scores
		scores = z.new_zeros(size=(batch_size*self.beam_width,self.output_size))
		# an array of numbers for displacement ie. if batch_size is 2 and beam_width is 3 then this is [0,0,0,3,3,3]. This is used later for indexing
		beam_displacement = torch.arange(start=0,end=batch_size*self.beam_width,step=self.beam_width,dtype=torch.long,device=z.device).view(-1,1).repeat(1,self.beam_width).view(-1)
		for i in range(num_steps):
			input = self.activation(self.embedding(previous_output))
			step_input = torch.cat([input,z],dim=1)
			step_input = self.batchnorm2(step_input)
			step_mem = memory[:,i,:] # step_mem is of size [batch,step_input_size]
			out,hx,hm = self.memcell(step_input,step_mem,hx=hx,hm=hm)
			out = self.activation(h)
			out = self.batchnorm3(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			# compute new scores
			next_scores = scores + torch.log(out+1e-8)
			# select top-k scores where k is the beam width
			score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
			# flatten the output
			outputs = outputs.view(-1)
			# get the indices in the original onehot output by finding the module of the vocab size
			indices = torch.fmod(outputs,self.output_size)
			# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
			beam_indices = torch.div(outputs,self.output_size) + beam_displacement
			# check if some elements/words are repeated
			res = torch.eq(previous_output,indices).nonzero()
			# some elements/words is repeated
			retries = 0
			while res.shape[0] > 0:
				mask = torch.ones(size=(batch_size*self.beam_width,self.output_size),requires_grad=False,device=z.device)
				# set the mask to be zero when an option is non selectable
				mask[beam_indices[res],indices[res]] = 0
				# apply the mask
				out = out * mask
				# set the score for the repeated elements to be low
				next_scores = scores + torch.log(out+1e-8)
				# select top-k scores where k is the beam width
				score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
				# flatten the output
				outputs = outputs.view(-1)
				# get the indices in the original onehot output by finding the module of the vocab size
				indices = torch.fmod(outputs,self.output_size)
				# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
				beam_indices = torch.div(outputs,self.output_size) + beam_displacement
				# check if some elements/words are repeated
				res = torch.eq(previous_output,indices).nonzero()
				if retries > 10:
					break
				retries += 1
			# copy the score for each selected candidate
			scores = score.view(-1,1).repeat(1,self.output_size)
			# renormalize the output
			out = out/out.sum(-1).view(-1,1).repeat(1,self.output_size)
			# append the prediction to output
			predictions.append(out[beam_indices,:])
			# detach the output such that we don't backpropagate through timesteps
			previous_output = indices.detach()
		output = torch.stack(predictions).transpose(1,0)
		# initialize an output_mask such that we can filter out sentences
		output_mask = torch.zeros_like(output)
		# set the selected sentences output_mask to 1
		output_mask[scores[:,0].view(batch_size,-1).argmax(dim=-1) + beam_displacement.view(batch_size,-1)[:,0]] = 1
		# collect the best prediction for each sample in batch
		output = (output*output_mask).view(batch_size,self.beam_width,num_steps,self.output_size)
		# sum the beam sentences. Since the sentences that is not selected is zero this doesn't change the actual sentences
		output = output.sum(1)
		return output

	def forward_greedy(self,z,num_steps,temperature,x=None):
		if num_steps > self.max_seq_len:
			raise ValueError("num_steps ({}) must be less or equal to max_seq_len ({})".format(num_steps,self.max_seq_len))
		predictions = []
		z = self.batchnorm1(self.activation(self.z2h(z)))
		hx = z # initialize the hidden state
		hm = z # initialize the hidden state for memory
		memory = self.memory
		memory = memory[:num_steps,:]
		memory = memory.expand(z.size(0),-1,-1) # copy the memory for each batch position
		previous_output = z.new_zeros(size=(z.size(0),),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		# run rnn
		for i in range(num_steps):
			# for the input
			input = self.activation(self.embedding(previous_output)) # previous_output is of size [batch,hidden_size]
			step_input = torch.cat([input,z],dim=1) # step_input is of size [batch,step_input_size]
			step_input = self.batchnorm2(step_input)
			step_mem = memory[:,i,:] # step_mem is of size [batch,step_input_size]
			out,hx,hm = self.memcell(step_input,step_mem,hx=hx,hm=hm)
			out = self.activation(out)
			out = self.batchnorm3(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out,dim=-1)
				previous_output = previous_output.detach()
			predictions.append(out)			
		output = torch.stack(predictions).transpose(1,0)
		return output

class TransformerGenerator(nn.Module):
	def __init__(self,hidden_size,num_heads,noise_size,output_size,num_layers,max_seq_len,d_ff=2048,activation=nn.LeakyReLU(0.2),SOS_TOKEN=None,PAD_TOKEN=None,beam_width=1):
		super().__init__()
		# internal variable sizes
		self.hidden_size = hidden_size
		self.output_size = output_size
		step_input_size = hidden_size + hidden_size
		self.PAD_TOKEN = PAD_TOKEN if PAD_TOKEN is not None else output_size-2
		self.SOS_TOKEN = SOS_TOKEN if SOS_TOKEN is not None else output_size-1
		# layer definitions
		self.z2h = nn.Linear(noise_size,hidden_size)
		self.embedding = nn.Embedding(output_size,hidden_size,padding_idx=PAD_TOKEN)
		self.pos_embedding = PositionalEmbedding(max_seq_len+1,hidden_size)
		self.s2h = nn.Linear(step_input_size,hidden_size)
		self.transformer = TransformerDecoder(output_size,num_layers,hidden_size,num_heads,d_ff=d_ff,dropout_prob=0.1)
		self.h2o = nn.Linear(hidden_size,output_size)
		self.activation = activation
		self.last_activation = GumbelSoftmax()
		self.beam_width = beam_width

	def forward(self,z,num_steps,temperature,x=None):
		if self.beam_width > 1 and x is None:
			return self.forward_beam(z=z,num_steps=num_steps,temperature=temperature)
		else:
			return self.forward_greedy(z=z,num_steps=num_steps,temperature=temperature,x=x)

	def forward_beam(self,z,num_steps,temperature):
		batch_size = z.size(0)
		input = z.new_zeros(size=(batch_size*self.beam_width,num_steps),dtype=torch.long,requires_grad=False)
		input[:,:] = self.PAD_TOKEN
		previous_output = z.new_zeros(size=(batch_size*self.beam_width,),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		z = self.activation(self.z2h(z)).view(batch_size,1,-1).repeat(self.beam_width,num_steps,1)
		# a table for storing the scores
		scores = z.new_zeros(size=(batch_size*self.beam_width,self.output_size))
		# an array of numbers for displacement ie. if batch_size is 2 and beam_width is 3 then this is [0,0,0,3,3,3]. This is used later for indexing
		beam_displacement = torch.arange(start=0,end=batch_size*self.beam_width,step=self.beam_width,dtype=torch.long,device=z.device).view(-1,1).repeat(1,self.beam_width).view(-1)
		for i in range(num_steps):
			input[:,i] = previous_output
			step_input = self.embedding(input)
			step_input = self.pos_embedding(step_input)
			step_input = torch.cat([step_input,z],dim=2) # step_input is of size [batch,seq_len,step_input_size]
			step_input = self.activation(self.s2h(step_input))
			mask = create_target_mask(input,self.PAD_TOKEN)
			out = self.transformer(step_input,mask=mask)
			#out = torch.index_select(out,dim=1,index=torch.tensor([i],requires_grad=False,device=z.device)).squeeze(1) # out[:,i,:]
			out = self.activation(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			# compute new scores
			next_scores = scores + torch.log(out+1e-8)
			# select top-k scores where k is the beam width
			score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
			# flatten the output
			outputs = outputs.view(-1)
			# get the indices in the original onehot output by finding the module of the vocab size
			indices = torch.fmod(outputs,self.output_size)
			# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
			beam_indices = torch.div(outputs,self.output_size) + beam_displacement
			# check if some elements/words are repeated
			res = torch.eq(previous_output,indices).nonzero()
			# some elements/words is repeated
			retries = 0
			while res.shape[0] > 0:
				mask = torch.ones(size=(batch_size*self.beam_width,self.output_size),requires_grad=False,device=z.device)
				# set the mask to be zero when an option is non selectable
				mask[beam_indices[res],indices[res]] = 0
				# apply the mask
				out = out * mask
				# set the score for the repeated elements to be low
				next_scores = scores + torch.log(out+1e-8)
				# select top-k scores where k is the beam width
				score,outputs = next_scores.view(batch_size,-1).topk(self.beam_width,dim=-1)
				# flatten the output
				outputs = outputs.view(-1)
				# get the indices in the original onehot output by finding the module of the vocab size
				indices = torch.fmod(outputs,self.output_size)
				# find the index in the beam/batch for the onehot output. Add beam displacement to get correct index
				beam_indices = torch.div(outputs,self.output_size) + beam_displacement
				# check if some elements/words are repeated
				res = torch.eq(previous_output,indices).nonzero()
				if retries > 10:
					break
				retries += 1
			# copy the score for each selected candidate
			scores = score.view(-1,1).repeat(1,self.output_size)
			# renormalize the output
			out = out/out.sum(-1).view(-1,1).repeat(1,self.output_size)
			# append the prediction to output
			#predictions.append(out[beam_indices,:])
			# detach the output such that we don't backpropagate through timesteps
			previous_output = indices.detach()
		output = torch.stack(predictions).transpose(1,0)
		# initialize an output_mask such that we can filter out sentences
		output_mask = torch.zeros_like(output)
		# set the selected sentences output_mask to 1
		output_mask[scores[:,0].view(batch_size,-1).argmax(dim=-1) + beam_displacement.view(batch_size,-1)[:,0]] = 1
		# collect the best prediction for each sample in batch
		output = (output*output_mask).view(batch_size,self.beam_width,num_steps,self.output_size)
		# sum the beam sentences. Since the sentences that is not selected is zero this doesn't change the actual sentences
		output = output.sum(1)
		return output

	def forward_greedy(self,z,num_steps,temperature,x=None):
		batch_size = z.size(0)
		input = z.new_zeros(size=(batch_size,num_steps),dtype=torch.long,requires_grad=False)
		input[:,:] = self.PAD_TOKEN
		previous_output = z.new_zeros(size=(z.size(0),),dtype=torch.long)
		previous_output[:] = self.SOS_TOKEN # <sos> token
		z = self.activation(self.z2h(z)).view(batch_size,1,-1).repeat(1,num_steps,1)
		for i in range(num_steps):
			input[:,i] = previous_output
			step_input = self.embedding(input)
			step_input = self.pos_embedding(step_input)
			step_input = torch.cat([step_input,z],dim=2) # step_input is of size [batch,seq_len,step_input_size]
			step_input = self.activation(self.s2h(step_input))
			mask = create_target_mask(input,self.PAD_TOKEN)
			out = self.transformer(step_input,mask=mask)
			#out = torch.index_select(out,dim=1,index=torch.tensor([i],device=z.device)).squeeze(1) # out[:,i,:]
			out = self.activation(out)
			out = self.h2o(out)
			out = self.last_activation(out,temperature)
			if x is not None: # teacher forcing
				previous_output = x[:,i]
			else: # use prediction as input
				previous_output = torch.argmax(out[:,i,:],dim=-1)
				previous_output = previous_output.detach()
		output = out
		return output

if __name__ == '__main__':
	batch_size = 2
	num_steps = 3
	noise_size = 1
	output_size = 4
	z = torch.randn(batch_size,noise_size)
	model = TransformerGenerator(hidden_size=8,num_heads=1,noise_size=noise_size,output_size=output_size,
		num_layers=4,max_seq_len=num_steps,d_ff=128,SOS_TOKEN=1,PAD_TOKEN=0,beam_width=1)
	out = model(z,num_steps,temperature=1.0)
	#print(out.argmax(dim=-1))
