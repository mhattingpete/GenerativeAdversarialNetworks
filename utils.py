import torch

def img2vec(img):
	'''
	Converts an image into a vector
	'''
	return img.view(img.size(0),-1)

def vec2img(vec,image_size):
	'''
	Converts a vector into an image
	'''
	return vec.view(vec.size(0),1,image_size,image_size)

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
	Tensor of ones to match the fake target
	'''
	return torch.zeros(num_samples,1).to(device)
	