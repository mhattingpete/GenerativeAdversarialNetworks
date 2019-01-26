import torch
import os
import errno
from torch import nn,autograd
import numpy as np

def tensor_to_list_of_words(batch,num_to_word_vocab):
	text_translated = []
	for line in batch:
		line_translated = []
		for word in line:
			word_translated = num_to_word_vocab[word.cpu().numpy().tolist()]
			if word_translated in ["<pad>"]:
				continue
			line_translated.append(word_translated)
			if word_translated == "<eos>":
				break
		text_translated.append(line_translated)
	return text_translated

def save_model(model,model_save_path):
	create_dir(model_save_path)
	torch.save(model.state_dict(),os.path.join(model_save_path,model.__class__.__name__))

def load_model(model,model_load_path):
	try:
		model.load_state_dict(torch.load(model_load_path))
	except Exception:
		raise

def create_dir(path):
	try:
		os.mkdir(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def img2vec(img):
	'''
	Converts an image into a vector
	'''
	return img.view(img.size(0),-1)

def vec2img(vec,image_size):
	'''
	Converts a vector into an image
	'''
	return vec.view(vec.size(0),-1,image_size,image_size)

def sample_noise(num_samples,noise_dim,device):
	'''
	Sample a vector of normal distributed values
	'''
	return torch.randn(num_samples,noise_dim).to(device)

def true_target(num_samples,device):
	'''
	Tensor of ones to match the true target
	'''
	return torch.ones(num_samples,1).to(device)

def fake_target(num_samples,device):
	'''
	Tensor of zeros to match the fake target
	'''
	return torch.zeros(num_samples,1).to(device)
	
def onehot(vec,output_size):
	out_size = vec.size()+torch.Size([output_size])
	vec = vec.view(-1,1)
	out = vec.new_zeros(size=(vec.size(0),output_size),dtype=torch.long)
	out.scatter_(1,vec.long(),1)
	return out.view(out_size).float()

def num_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_non_pad_mask(seq,PAD_TOKEN):
	return seq.ne(PAD_TOKEN).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq,PAD_TOKEN):
	"""
	For masking out padding part of sequence
	"""
	seq_len = seq.size(1)
	padding_mask = seq.eq(PAD_TOKEN)
	padding_mask = padding_mask.unsqueeze(1).expand(-1,seq_len,-1) # size [batch_size x seq_len x seq_len]
	return padding_mask

def get_subsequent_mask(seq):
	"""
	For preventing lookahead
	"""
	batch_size,seq_len = seq.size()
	subsequent_mask = torch.triu(torch.ones((seq_len,seq_len),device=seq.device,dtype=torch.uint8),diagonal=1)
	subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size,-1,-1) # size [batch_size x seq_len x seq_len]
	return subsequent_mask

class RaSGANLoss: 
	# Relativistic Standard GAN
	def __init__(self):
		self.loss_fun = nn.BCEWithLogitsLoss()

	def __call__(self,input,opposite,target_is_real):
		N = input.size(0)
		if target_is_real:
			target = true_target(N,input.device)
		else:
			target = fake_target(N,input.device)
		return self.loss_fun(input-torch.mean(opposite,dim=0),target)

class WGAN_GPLoss:
	# Wasserstein GP GAN
	def __init__(self,discriminator):
		self.discriminator = discriminator
		self.LAMBDA = 10

	def generator_loss(self,pred_fake):
		return -pred_fake.mean()

	def discriminator_loss(self,pred_real,pred_fake,real_data,fake_data):
		batch_size = real_data.size(0)
		penalty = self.calc_gradient_penalty(real_data.contiguous().view(batch_size,-1),fake_data.contiguous().view(batch_size,-1),real_data.size())
		return pred_fake.mean() - pred_real.mean() + penalty

	def calc_gradient_penalty(self,real_data,fake_data,original_size):
		batch_size = real_data.size(0)
		alpha = torch.rand(batch_size,1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.to(real_data.device)
		
		interpolates = alpha * real_data + ((1 - alpha) * fake_data)
		interpolates = autograd.Variable(interpolates,requires_grad=True)
		interpolates = interpolates.to(real_data.device)
		
		disc_interpolates = self.discriminator(interpolates.view(original_size)).view(-1)

		gradients = autograd.grad(outputs=disc_interpolates,inputs=interpolates,
			grad_outputs=torch.ones(batch_size).to(real_data.device),
			create_graph=True,retain_graph=True,only_inputs=True)[0]
		gradient_penalty = ((gradients.norm(p=2,dim=1)-1)**2).mean()*self.LAMBDA
		return gradient_penalty
