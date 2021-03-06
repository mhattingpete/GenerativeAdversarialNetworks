import sys
sys.path.append("../../Scripts")

import torch
import os
import json

with open(sys.argv[1],"rb") as f:
	config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

import numpy as np
from torch import nn,optim
from tensorboardX import SummaryWriter
import random
from nltk.translate.bleu_score import sentence_bleu

from utils import onehot,num_parameters,sample_noise,save_model,load_model,tensor_to_list_of_words,RaSGANLoss,WGAN_GPLoss
from layers import GumbelSoftmax
from visualize import tensor_to_words
import generators
import discriminators

from Dataloaders.load_coco_captions_dataset import create_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = os.path.join("../Saved_models",config["model_config"]["save_name"])

index = 0
if not os.path.exists(os.path.join(model_save_path,"pretrain")):
        os.mkdir(os.path.join(model_save_path,"pretrain"))
pre_summary_path = os.path.join(model_save_path,"pretrain/exp-{}".format(index))
while os.path.exists(pre_summary_path):
	index += 1
	pre_summary_path = os.path.join(model_save_path,"pretrain/exp-{}".format(index))
pretrain_dis = SummaryWriter(pre_summary_path)

index = 0
if not os.path.exists(os.path.join(model_save_path,"train")):
	os.mkdir(os.path.join(model_save_path,"train"))
summary_path = os.path.join(model_save_path,"train/exp-{}".format(index))
while os.path.exists(summary_path):
	index += 1
	summary_path = os.path.join(model_save_path,"train/exp-{}".format(index))
dis = SummaryWriter(summary_path)

print("Does model save path exist:",os.path.exists(model_save_path))

print(device)

batch_size = config["model_config"]["batch_size"]

min_vocab_freq = 1 if "min_vocab_freq" not in config["dataset_config"] else config["dataset_config"]["min_vocab_freq"]
max_vocab_size = None if "max_vocab_size" not in config["dataset_config"] else config["dataset_config"]["max_vocab_size"]


"""
Load dataset and prepare for iteration
"""
data_dict = create_dataset(config["dataset_config"]["path"],batch_size,min_vocab_freq=min_vocab_freq,max_vocab_size=max_vocab_size)
train_data,val_data,test_data = data_dict["data_iters"]
text_field = data_dict["fields"]
num_classes = data_dict["num_classes"]
SOS_TOKEN,EOS_TOKEN,UNK_TOKEN,PAD_TOKEN = data_dict["tokens"]
max_seq_len = data_dict["max_seq_len"]
num_to_word_vocab = data_dict["num_to_word_vocab"]
word_to_num_vocab = data_dict["word_to_num_vocab"]
train_iter = iter(train_data)
val_iter = iter(val_data)
test_iter = iter(train_data)


"""
Prepare models
"""
g_pre_lr = config["model_config"]["generator"]["pre_lr"]
g_lr = config["model_config"]["generator"]["lr"]
d_lr = config["model_config"]["discriminator"]["lr"]

noise_size = config["model_config"]["noise_size"]

num_test_samples = 100
test_noise = sample_noise(num_test_samples,noise_size,device)

vocab = word_to_num_vocab.keys() if config["model_config"]["use_glove"].lower() == "true" else None

# intialize models
if "hidden_size" not in config["model_config"]["generator"] and "mem_slots" in config["model_config"]["generator"] and \
"head_size" in config["model_config"]["generator"] and "num_heads" in config["model_config"]["generator"]:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(mem_slots=config["model_config"]["generator"]["mem_slots"],
		head_size=config["model_config"]["generator"]["head_size"],num_heads=config["model_config"]["generator"]["num_heads"],
		noise_size=noise_size,output_size=num_classes,vocab=vocab,SOS_TOKEN=SOS_TOKEN,beam_width=config["model_config"]["generator"]["beam_width"]).to(device)
elif "hidden_size" in config["model_config"]["generator"] and "sim_size" in config["model_config"]["generator"] and \
"similarity" in config["model_config"]["generator"]:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(hidden_size=config["model_config"]["generator"]["hidden_size"],
		noise_size=noise_size,output_size=num_classes,max_seq_len=max_seq_len,sim_size=config["model_config"]["generator"]["sim_size"],
		similarity=getattr(nn,config["model_config"]["generator"]["similarity"])(dim=-1),vocab=vocab,SOS_TOKEN=SOS_TOKEN,beam_width=config["model_config"]["generator"]["beam_width"]).to(device)
elif "TransformerGenerator" in config["model_config"]["generator"]["name"]:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(hidden_size=config["model_config"]["generator"]["hidden_size"],
		num_heads=config["model_config"]["generator"]["num_heads"],noise_size=noise_size,output_size=num_classes,
		num_layers=config["model_config"]["generator"]["num_layers"],max_seq_len=max_seq_len,d_ff=config["model_config"]["generator"]["d_ff"],
		vocab=vocab,SOS_TOKEN=SOS_TOKEN,PAD_TOKEN=PAD_TOKEN,beam_width=config["model_config"]["generator"]["beam_width"]).to(device)
else:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(hidden_size=config["model_config"]["generator"]["hidden_size"],
		noise_size=noise_size,output_size=num_classes,vocab=vocab,SOS_TOKEN=SOS_TOKEN,beam_width=config["model_config"]["generator"]["beam_width"]).to(device)

multiple_embeddings = "num_embeddings" in config["model_config"]["discriminator"]
if multiple_embeddings:
	discriminator = getattr(discriminators,config["model_config"]["discriminator"]["name"])(input_size=num_classes,
	hidden_size=config["model_config"]["discriminator"]["hidden_size"],rnn_size=config["model_config"]["discriminator"]["rnn_size"],output_size=1,
	num_embeddings=config["model_config"]["discriminator"]["num_embeddings"]).to(device)
elif "hidden_size" in config["model_config"]["discriminator"] and "sim_size" in config["model_config"]["discriminator"] and \
"similarity" in config["model_config"]["discriminator"]:
	discriminator = getattr(discriminators,config["model_config"]["discriminator"]["name"])(input_size=num_classes,
	hidden_size=config["model_config"]["discriminator"]["hidden_size"],output_size=1,
	max_seq_len=max_seq_len,sim_size=config["model_config"]["discriminator"]["sim_size"],
	similarity=getattr(nn,config["model_config"]["discriminator"]["similarity"])(dim=-1)).to(device)
else:
	discriminator = getattr(discriminators,config["model_config"]["discriminator"]["name"])(input_size=num_classes,
	hidden_size=config["model_config"]["discriminator"]["hidden_size"],output_size=1).to(device)

def weight_init(m):
	if isinstance(m,nn.Linear) or isinstance(m,nn.Conv1d):
		nn.init.xavier_uniform_(m.weight.data)
	elif isinstance(m,nn.GRUCell):
		nn.init.xavier_uniform_(m.weight_ih.data)
		nn.init.xavier_uniform_(m.weight_hh.data)

generator.apply(weight_init)
discriminator.apply(weight_init)

# otpimizers
g_pre_optimizer = getattr(optim,config["model_config"]["generator"]["optimizer"])(generator.parameters(),lr=g_pre_lr,amsgrad=True)
g_optimizer = getattr(optim,config["model_config"]["generator"]["optimizer"])(generator.parameters(),lr=g_lr,amsgrad=True)
d_optimizer = getattr(optim,config["model_config"]["discriminator"]["optimizer"])(discriminator.parameters(),lr=d_lr,weight_decay=1e-4,amsgrad=True)
use_g_lr_scheduler = "lr_scheduler" in config["model_config"]["generator"]
use_d_lr_scheduler = "lr_scheduler" in config["model_config"]["discriminator"]

if use_g_lr_scheduler:
	pre_g_lr_scheduler = getattr(optim.lr_scheduler,config["model_config"]["generator"]["pre_lr_scheduler"]["name"])(g_pre_optimizer,
		mode=config["model_config"]["generator"]["pre_lr_scheduler"]["mode"],factor=config["model_config"]["generator"]["pre_lr_scheduler"]["factor"],
		patience=config["model_config"]["generator"]["pre_lr_scheduler"]["patience"],verbose=True)
g_lr_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer,config["model_config"]["generator"]["lr_scheduler"]["factor"])
d_lr_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer,config["model_config"]["discriminator"]["lr_scheduler"]["factor"])

# losses
supported_losses = ["RaSGAN","WGAN-GP"]
loss_name = config["train_config"]["loss_fun"] if config["train_config"]["loss_fun"] in supported_losses else "WGAN-GP"
if loss_name == "RaSGAN": # RaSGAN
	loss_fun = RaSGANLoss()
else: # WGAN-GP
	loss_fun = WGAN_GPLoss(discriminator)

loss_weight = torch.ones(num_classes).to(device)
loss_weight[SOS_TOKEN] = 0.0
#loss_weight[EOS_TOKEN] = 0.0
#loss_weight[UNK_TOKEN] = 0.0
loss_weight[PAD_TOKEN] = 0.0
pretrain_loss_fun = nn.NLLLoss(weight=loss_weight)

np_g = num_parameters(generator)
np_d = num_parameters(discriminator)
print("Number of parameters for G: {}\nNumber of parameters for D: {}\nNumber of parameters in total: {}"
	  .format(np_g,np_d,np_g+np_d))

def pretrain_generator(real_data,fake_data,optimizer):
	'''
	Pretrain the generator to generate realistic samples for a good initialization
	'''
	# Reset gradients
	optimizer.zero_grad()
	loss = 0
	fake_data = torch.log(fake_data+1e-8)
	for i in range(fake_data.size(1)):
		loss += pretrain_loss_fun(fake_data[:,i,:],real_data[:,i])
	loss /= fake_data.size(1)
	loss.backward()
	optimizer.step()
	return loss

def train_generator(discriminator,real_data_onehot,fake_data,optimizer):
	'''
	Train the generator to generate realistic samples and thereby fool the discriminator
	'''
	N = fake_data.size(0)
	# Reset gradients
	optimizer.zero_grad()
	if loss_name == "RaSGAN":
		# 1.1 Train on Real Data
		pred_real = discriminator(real_data_onehot)
	# 1.2 Train on Fake Data
	pred_fake = discriminator(fake_data)
	if multiple_embeddings:
		losses = []
		for i in range(pred_real.size(1)):
			if loss_name == "RaSGAN":
				losses.append(loss_fun(input=pred_fake[:,i,:],opposite=pred_real[:,i,:],target_is_real=True) + \
					loss_fun(input=pred_real[:,i,:],opposite=pred_fake[:,i,:],target_is_real=False))
			else: # WGAN-GP
				losses.append(loss_fun.generator_loss(pred_fake[:,i,:]))
		loss = torch.stack(losses).mean()
	else:
		if loss_name == "RaSGAN":
			loss = loss_fun(input=pred_fake,opposite=pred_real,target_is_real=True) + loss_fun(input=pred_real,opposite=pred_fake,target_is_real=False)
		else: # WGAN-GP
			loss = loss_fun.generator_loss(pred_fake)
	loss.backward()
	optimizer.step()
	return loss

def train_discriminator(discriminator,real_data_onehot,fake_data,optimizer):
	'''
	Train the discriminator to distinguish between real and fake data
	'''
	N = real_data_onehot.size(0)
	# Reset gradients
	optimizer.zero_grad()
	# 1.1 Train on Real Data
	pred_real = discriminator(real_data_onehot)
	# 1.2 Train on Fake Data
	pred_fake = discriminator(fake_data)
	if multiple_embeddings:
		losses = []
		for i in range(pred_real.size(1)):
			if loss_name == "RaSGAN":
				losses.append(loss_fun(input=pred_real[:,i,:],opposite=pred_fake[:,i,:],target_is_real=True) + \
					loss_fun(input=pred_fake[:,i,:],opposite=pred_real[:,i,:],target_is_real=False))
			else: # WGAN-GP
				losses.append(loss_fun.discriminator_loss(pred_real[:,i,:],pred_fake[:,i,:],real_data_onehot[:,i,:],fake_data[:,i,:]))
		loss = torch.stack(losses).mean()
	else:
		if loss_name == "RaSGAN":
			loss = loss_fun(input=pred_real,opposite=pred_fake,target_is_real=True) + loss_fun(input=pred_fake,opposite=pred_real,target_is_real=False)
		else: # WGAN-GP
			loss = loss_fun.discriminator_loss(pred_real,pred_fake,real_data_onehot,fake_data)
	loss.backward()
	nn.utils.clip_grad_norm_(discriminator.parameters(),max_norm=100,norm_type=2)
	# 1.3 Update weights with gradients
	optimizer.step()
	return loss

gen_steps = config["train_config"]["gen_steps"]
gen_train_freq = config["train_config"]["gen_train_freq"]
max_temperature = torch.FloatTensor([config["train_config"]["max_temperature"]]).to(device)
pretrain_temperature = torch.FloatTensor([config["train_config"]["pretrain_temperature"]]).to(device)

epochs_pretrain = config["train_config"]["epochs_pretrain"]
pretrain_step = 0

num_epochs = config["train_config"]["num_epochs"]
global_step = 0
epoch = 0
d_error = 0
g_error = 0

pre_text_log = open(os.path.join(pre_summary_path,"log.txt"),"a")

# pretrain generator
for ep in range(epochs_pretrain):
	train_iter = iter(train_data)
	batch_error = []
	for n_batch,batch in enumerate(train_iter):
		real_data = batch.text.to(device)
		N = real_data.size(0)
		num_steps = real_data.size(1)

		# Generate fake data
		noise = sample_noise(N,noise_size,device)
		fake_data = generator(z=noise,num_steps=num_steps,temperature=pretrain_temperature,
							  x=real_data.long())
		# Train G
		pretrain_g_error = pretrain_generator(real_data,fake_data,g_pre_optimizer)
		batch_error.append(pretrain_g_error)
		
		# Log batch error and delete tensors
		pretrain_step += 1
		if pretrain_step % 200 == 0:
			pretrain_dis.add_scalar("pretrain epoch",ep,pretrain_step)
			pretrain_dis.add_scalar("pretrain_g_error",pretrain_g_error.item(),pretrain_step)
	if use_g_lr_scheduler:
		pre_g_lr_scheduler.step(torch.stack(batch_error).mean())
	if ep % 5 == 0:
		test_samples = generator(z=test_noise,num_steps=num_steps,temperature=pretrain_temperature)
		test_samples_vals = torch.argmax(test_samples,dim=2)
		test_samples_text = tensor_to_words(test_samples_vals,num_to_word_vocab)
		pre_text_log.write("Epoch: "+str(ep)+"\n"+test_samples_text+"\n")

pre_text_log.close()
text_log = open(os.path.join(summary_path,"log.txt"),"a")

softmax = GumbelSoftmax()

# train adverserially
try:
	while epoch < num_epochs:
		train_iter = iter(train_data)
		temperature = max_temperature**((epoch+1)/num_epochs)
		g_lr_scheduler.step()
		d_lr_scheduler.step()
		for n_batch,batch in enumerate(train_iter):
			real_data = batch.text.to(device)
			N = real_data.size(0)
			num_steps = real_data.size(1)
			# 1. Train Discriminator
			real_data_onehot = onehot(real_data,num_classes)
			real_data_onehot[real_data_onehot==1] = 0.7
			real_data_onehot[real_data_onehot==0] = (1.0-0.7)/(num_classes-1.0)
			real_data_onehot = softmax(real_data_onehot,temperature)

			# Generate fake data and detach 
			# (so gradients are not calculated for generator)
			noise_tensor = sample_noise(N,noise_size,device)
			with torch.no_grad():
				fake_data = generator(z=noise_tensor,num_steps=num_steps,temperature=temperature).detach()
			# Train D
			d_error = train_discriminator(discriminator,real_data_onehot,fake_data,d_optimizer)

			# 2. Train Generator every 'gen_train_freq' steps
			if global_step % gen_train_freq == 0:
				for _ in range(gen_steps):
					# Generate fake data
					noise_tensor = sample_noise(N,noise_size,device)
					fake_data = generator(z=noise_tensor,num_steps=num_steps,temperature=temperature)
					# Train G
					g_error = train_generator(discriminator,real_data_onehot,fake_data,g_optimizer)
					g_error = g_error.item()
			global_step += 1

			# Display Progress every few batches
			if global_step % 50 == 0:
				dis.add_scalar("epoch",epoch,global_step)
				dis.add_scalar("g_error",g_error,global_step)
				dis.add_scalar("d_error",d_error.item(),global_step)
				dis.add_scalar("beta",temperature.item(),global_step)
		if epoch % 20 == 0:
			test_samples = generator(z=test_noise,num_steps=num_steps,temperature=temperature)
			test_samples_vals = torch.argmax(test_samples,dim=2)
			test_samples_text = tensor_to_words(test_samples_vals,num_to_word_vocab)
			text_log.write("Epoch: "+str(epoch)+"\n"+test_samples_text+"\n")
		if epoch % 10 == 0:
			save_model(generator,summary_path)
			save_model(discriminator,summary_path)
		epoch += 1
except:
	save_model(generator,summary_path)
	save_model(discriminator,summary_path)

test_samples = generator(z=test_noise,num_steps=num_steps,temperature=temperature)
test_samples_vals = torch.argmax(test_samples,dim=2)
test_samples_text = tensor_to_words(test_samples_vals,num_to_word_vocab)
text_log.write("After training:\n"+test_samples_text+"\n")

def nll_gen(real_data,fake_data):
	'''
	Evaluate the generators ability to generate diverse samples
	'''    
	loss = 0
	fake_data = torch.log(fake_data+1e-8)
	for i in range(fake_data.size(1)):
		loss += pretrain_loss_fun(fake_data[:,i,:],real_data[:,i])
	loss /= fake_data.size(1)
	return loss

# evaluate generator
nll_gen_error = []
hypothesis_list = []
reference = []
for n_batch,batch in enumerate(val_iter):
	real_data = batch.text.to(device)
	N = real_data.size(0)
	num_steps = real_data.size(1)

	# Generate fake data
	noise = sample_noise(N,noise_size,device)
	fake_data = generator(z=noise,num_steps=num_steps,temperature=max_temperature,
						  x=real_data.long())
	# Calculate nll_gen
	nll_g_error = nll_gen(real_data,fake_data)
	nll_gen_error.append(nll_g_error.item())

	# Save sentences for bleu score calculation
	fake_data = generator(z=noise,num_steps=num_steps,temperature=max_temperature)
	fake_data_vals = torch.argmax(fake_data,dim=2)
	fake_data_text = tensor_to_list_of_words(fake_data_vals,num_to_word_vocab)
	real_data_text = tensor_to_list_of_words(real_data,num_to_word_vocab)
	hypothesis_list.extend(fake_data_text)
	reference.extend(real_data_text)

nll_gen_error = np.array(nll_gen_error)
nll_gen_error_mean = nll_gen_error.mean()
print(nll_gen_error_mean)

random.shuffle(hypothesis_list)
random.shuffle(reference)
reference = reference[:5000]
n_gram_bleu_scores = {"{}-gram".format(gram):0 for gram in range(2,6)}
for ngram in range(2,6):
	weight = tuple((1./ngram for _ in range(ngram)))
	bleu_score = []
	for h in hypothesis_list[:2000]:
		BLEUscore = sentence_bleu(reference,h,weight)
		bleu_score.append(BLEUscore)
	current_bleu = 1.0*sum(bleu_score)/len(bleu_score)
	n_gram_bleu_scores["{}-gram".format(len(weight))] = current_bleu
	if current_bleu < 1e-2:
		break

text_log.write("\n\nGot nll_gen mean: {}".format(nll_gen_error_mean))
for gram,score in n_gram_bleu_scores.items():
	text_log.write("\nGot {} score: {}".format(gram,score))

text_log.close()

save_model(generator,summary_path)
save_model(discriminator,summary_path)

# to load model run
#load_model(model,model_load_path)
