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
	Tensor of ones to match the fake target
	'''
	return torch.zeros(num_samples,1).to(device)
	
def onehot(vec,output_size):
	out_size = vec.size()+torch.Size([output_size])
	vec = vec.view(-1,1)
	out = torch.zeros(vec.size(0),output_size).type(torch.LongTensor)
	out.scatter_(1,vec,1)
	return out.view(out_size)

def num_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
