import torch
import os
import errno

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
	Tensor of ones to match the fake target
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
