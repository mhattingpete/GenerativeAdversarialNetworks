import sys
sys.path.append("../../Scripts")

import torch
import os
import json

with open(sys.argv[1],"rb") as f:
	config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

from torch import nn,optim
from tensorboardX import SummaryWriter

from utils import onehot,num_parameters,sample_noise,true_target,fake_target
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
g_lr = config["model_config"]["generator"]["lr"]
d_lr = config["model_config"]["discriminator"]["lr"]

noise_size = config["model_config"]["noise_size"]

num_test_samples = 100
test_noise = sample_noise(num_test_samples,noise_size,device)

# intialize models
if "hidden_size" not in config["model_config"]["generator"] and "mem_slots" in config["model_config"]["generator"] and \
"head_size" in config["model_config"]["generator"] and "num_heads" in config["model_config"]["generator"]:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(mem_slots=config["model_config"]["generator"]["mem_slots"],
		head_size=config["model_config"]["generator"]["head_size"],num_heads=config["model_config"]["generator"]["num_heads"],
		noise_size=noise_size,output_size=num_classes,SOS_TOKEN=SOS_TOKEN).to(device)
elif "hidden_size" in config["model_config"]["generator"] and "num_heads" in config["model_config"]["generator"] and \
"similarity" in config["model_config"]["generator"]:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(hidden_size=config["model_config"]["generator"]["hidden_size"],
		noise_size=noise_size,output_size=num_classes,max_seq_len=max_seq_len,num_heads=config["model_config"]["generator"]["num_heads"],
		similarity=getattr(nn,generators,config["model_config"]["generator"]["similarity"])(dim=-1),SOS_TOKEN=SOS_TOKEN).to(device)
else:
	generator = getattr(generators,config["model_config"]["generator"]["name"])(hidden_size=config["model_config"]["generator"]["hidden_size"],
		noise_size=noise_size,output_size=num_classes,SOS_TOKEN=SOS_TOKEN).to(device)

multiple_embeddings = "num_embeddings" in config["model_config"]["discriminator"]
if multiple_embeddings:
	discriminator = getattr(discriminators,config["model_config"]["discriminator"]["name"])(input_size=num_classes,
	hidden_size=config["model_config"]["discriminator"]["hidden_size"],output_size=1,
	num_embeddings=config["model_config"]["discriminator"]["num_embeddings"]).to(device)
else:
	discriminator = getattr(discriminators,config["model_config"]["discriminator"]["name"])(input_size=num_classes,
	hidden_size=config["model_config"]["discriminator"]["hidden_size"],output_size=1).to(device)

# otpimizers
g_optimizer = getattr(optim,config["model_config"]["generator"]["optimizer"])(generator.parameters(),lr=g_lr)
d_optimizer = getattr(optim,config["model_config"]["discriminator"]["optimizer"])(discriminator.parameters(),lr=d_lr)
use_g_lr_scheduler = "lr_scheduler" in config["model_config"]["generator"]
use_d_lr_scheduler = "lr_scheduler" in config["model_config"]["discriminator"]

if use_g_lr_scheduler:
	g_lr_scheduler = getattr(optim.lr_scheduler,config["model_config"]["generator"]["lr_scheduler"]["name"])(g_optimizer,
		mode=config["model_config"]["generator"]["lr_scheduler"]["mode"],factor=config["model_config"]["generator"]["lr_scheduler"]["factor"],
		patience=config["model_config"]["generator"]["lr_scheduler"]["patience"],verbose=True)

# losses
loss_fun = nn.BCEWithLogitsLoss()
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
	num_classes = fake_data.size(2)
	# Reset gradients
	optimizer.zero_grad()
	loss = 0
	if fake_data.min() < 0:
		print(fake_data.min())
		assert False
	fake_data = torch.log(fake_data+1e-8)
	for i in range(fake_data.size(1)):
		loss += pretrain_loss_fun(fake_data[:,i,:],real_data[:,i])
	loss /= fake_data.size(1)
	loss.backward()
	optimizer.step()
	return loss

def train_generator(real_data_onehot,fake_data,optimizer):
	'''
	Train the generator to generate realistic samples and thereby fool the discriminator
	'''
	N = fake_data.size(0)
	# Reset gradients
	optimizer.zero_grad()
	# 1.1 Train on Real Data
	c_x_r = discriminator(real_data_onehot)
	# 1.2 Train on Fake Data
	fake_data_save = fake_data
	c_x_f = discriminator(fake_data)
	# compute the average of c_x_*
	c_x_r_mean = torch.mean(c_x_r,dim=0)
	c_x_f_mean = torch.mean(c_x_f,dim=0)
	if multiple_embeddings:
		losses_real = []
		losses_fake = []
		for i in range(c_x_r.size(1)):
			losses_real.append(loss_fun(c_x_r[:,i,:]-c_x_f_mean[i,:],fake_target(N,device)))
			losses_fake.append(loss_fun(c_x_f[:,i,:]-c_x_r_mean[i,:],true_target(N,device)))
		loss_real = torch.stack(losses_real).mean()
		loss_fake = torch.stack(losses_fake).mean()
	else:
		loss_real = loss_fun(c_x_r-c_x_f_mean,fake_target(N,device))
		loss_fake = loss_fun(c_x_f-c_x_r_mean,true_target(N,device))
	loss = (loss_real + loss_fake)/2.0
	if torch.isnan(loss):
		print(fake_data_save)
		assert False
	loss.backward()
	optimizer.step()
	return loss

def train_discriminator(real_data_onehot,fake_data,optimizer):
	'''
	Train the discriminator to distinguish between real and fake data
	'''
	N = real_data_onehot.size(0)
	# Reset gradients
	optimizer.zero_grad()
	# 1.1 Train on Real Data
	c_x_r = discriminator(real_data_onehot)
	# 1.2 Train on Fake Data
	fake_data_save = fake_data
	c_x_f = discriminator(fake_data)
	# compute the average of c_x_*
	c_x_r_mean = torch.mean(c_x_r,dim=0)
	c_x_f_mean = torch.mean(c_x_f,dim=0)
	if multiple_embeddings:
		losses_real = []
		losses_fake = []
		for i in range(c_x_r.size(1)):
			losses_real.append(loss_fun(c_x_r[:,i,:]-c_x_f_mean[i,:],true_target(N,device)))
			losses_fake.append(loss_fun(c_x_f[:,i,:]-c_x_r_mean[i,:],fake_target(N,device)))
		loss_real = torch.stack(losses_real).mean()
		loss_fake = torch.stack(losses_fake).mean()
	else:
		loss_real = loss_fun(c_x_r-c_x_f_mean,true_target(N,device))
		loss_fake = loss_fun(c_x_f-c_x_r_mean,fake_target(N,device))
	loss = (loss_real + loss_fake)/2.0
	if torch.isnan(loss):
		print(fake_data_save)
		assert False
	loss.backward()
	# 1.3 Update weights with gradients
	optimizer.step()
	return loss

gen_steps = 1
gen_train_freq = 1
max_temperature = torch.FloatTensor([config["train_config"]["max_temperature"]]).to(device)
pretrain_temperature = torch.FloatTensor([config["train_config"]["pretrain_temperature"]]).to(device)

epochs_pretrain = config["train_config"]["epochs_pretrain"]
pretrain_step = 0

num_epochs = config["train_config"]["num_epochs"]
global_step = 0
epoch = 0
d_error = 0
g_error = 0

# pretrain generator
for ep in range(epochs_pretrain):
	train_iter = iter(train_data)
	batch_error = []
	for n_batch,batch in enumerate(train_iter):
		real_data = batch.text.to(device)
		N = real_data.size(0)
		num_steps = real_data.size(1)
		real_data_onehot = onehot(real_data,num_classes)

		# Generate fake data
		generator.train()
		noise = sample_noise(N,noise_size,device)
		fake_data = generator(z=noise,num_steps=num_steps,temperature=pretrain_temperature,
							  x=real_data.long())
		if torch.isnan(fake_data).any():
			print(noise)
			assert False
		# Train G
		pretrain_g_error = pretrain_generator(real_data,fake_data,g_optimizer)
		batch_error.append(pretrain_g_error)
		
		# Log batch error and delete tensors
		#pretrain_dis.add_scalar("pretrain epoch",ep,pretrain_step)
		#pretrain_dis.add_scalar("pretrain_g_error",pretrain_g_error.item(),pretrain_step)
		pretrain_step += 1
		if pretrain_step % 200 == 0:
			pretrain_dis.add_scalar("pretrain epoch",ep,pretrain_step)
			pretrain_dis.add_scalar("pretrain_g_error",pretrain_g_error.item(),pretrain_step)
	if use_g_lr_scheduler:
		g_lr_scheduler.step(torch.stack(batch_error).mean())
	if ep % 10 == 0:
		generator.eval()
		test_samples = generator(z=test_noise,num_steps=num_steps,temperature=pretrain_temperature)
		test_samples_vals = torch.argmax(test_samples,dim=2)
		print(tensor_to_words(test_samples_vals,num_to_word_vocab))


log = open(os.path.join(model_save_path,"log.txt"),"a")

# train adverserially
while epoch < num_epochs:
	train_iter = iter(train_data)
	temperature = max_temperature**((epoch+1)/num_epochs)
	for n_batch,batch in enumerate(train_iter):
		real_data = batch.text.to(device)
		N = real_data.size(0)
		num_steps = real_data.size(1)
		# 1. Train Discriminator
		real_data_onehot = onehot(real_data,num_classes)
		real_data_onehot[real_data_onehot==1] = 0.9
		real_data_onehot[real_data_onehot==0] = (1.0-0.9)/(num_classes-1.0)

		# Generate fake data and detach 
		# (so gradients are not calculated for generator)
		noise_tensor = sample_noise(N,noise_size,device)
		with torch.no_grad():
			fake_data = generator(z=noise_tensor,num_steps=num_steps,temperature=temperature).detach()
		if torch.isnan(fake_data).any():
			print(noise_tensor)
			assert False
		# Train D
		d_error = train_discriminator(real_data_onehot,fake_data,d_optimizer)

		# 2. Train Generator every 'gen_train_freq' steps
		if global_step % gen_train_freq == 0:
			for _ in range(gen_steps):
				# Generate fake data
				noise_tensor = sample_noise(N,noise_size,device)
				fake_data = generator(z=noise_tensor,num_steps=num_steps,temperature=temperature)
				if torch.isnan(fake_data).any():
					print(noise_tensor)
					assert False
				# Train G
				g_error = train_generator(real_data_onehot,fake_data,g_optimizer)
				g_error = g_error.item()

		# Log batch error and delete tensors
		#dis.add_scalar("epoch",epoch,global_step)
		#dis.add_scalar("g_error",g_error,global_step)
		#dis.add_scalar("d_error",d_error.item(),global_step)
		#dis.add_scalar("beta",temperature.item(),global_step)
		global_step += 1

		# Display Progress every few batches
		if global_step % 50 == 0:
			dis.add_scalar("epoch",epoch,global_step)
			dis.add_scalar("g_error",g_error,global_step)
			dis.add_scalar("d_error",d_error.item(),global_step)
			dis.add_scalar("beta",temperature.item(),global_step)
			test_samples = generator(z=test_noise,num_steps=num_steps,temperature=temperature)
			test_samples_vals = torch.argmax(test_samples,dim=2)
			test_samples_text = tensor_to_words(test_samples_vals,num_to_word_vocab)
			print(test_samples_text)
			if epoch % 50 == 0:
				log.write("Epoch: "+str(epoch)+"\n"+test_samples_text+"\n")
	epoch += 1

def nll_gen(real_data,fake_data):
	'''
	Evaluate the generators ability to generate diverse samples
	'''    
	loss = 0
	fake_data = torch.log(fake_data)
	for i in range(fake_data.size(1)):
		loss += pretrain_loss_fun(fake_data[:,i,:],real_data[:,i])
	loss /= fake_data.size(1)
	return loss

#torch.cuda.empty_cache()

# evaluate generator
nll_gen_error = []
for n_batch,batch in enumerate(val_iter):
	real_data = batch.text
	N = real_data.size(0)
	num_steps = real_data.size(1)

	# Generate fake data
	generator.eval()
	noise = sample_noise(N,noise_size,device)
	fake_data = generator(z=noise,num_steps=num_steps,temperature=max_temperature,
						  x=real_data.long())
	# Train G
	nll_g_error = nll_gen(real_data,fake_data)
	nll_gen_error.append(nll_g_error)

nll_gen_error = torch.stack(nll_gen_error)
nll_gen_error_mean = nll_gen_error.mean()
print(nll_gen_error_mean)

log.write("After training got nll_gen mean: {}".format(nll_gen_error_mean))

log.close()

save_model(generator,model_save_path)
save_model(discriminator,model_save_path)

# to load model run
#load_model(model,model_load_path)
