import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from viz import updatable_display2

import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from generators import ConvGenerator
from discriminators import ConvDiscriminator
from utils import img2vec,vec2img,sample_noise,true_target,fake_target

def train_generator(noise,optimizer):
	'''
	Train the generator to generate realistic samples and thereby fool the discriminator
	'''
	N = noise.size(0)
	# Reset gradients
	optimizer.zero_grad()
	
	# Sample noise and generate fake data
	prediction = discriminator(noise).view(-1,1)
	loss = loss_fun(prediction,true_target(N,device))
	loss.backward()
	optimizer.step()
	return loss

def train_discriminator(real_data,fake_data,optimizer):
	'''
	Train the discriminator to distinguish between real and fake data
	'''
	N = real_data.size(0)
	# Reset gradients
	optimizer.zero_grad()

	# 1.1 Train on Real Data
	real_data += torch.normal(mean=torch.zeros(*real_data.shape))
	prediction_real = discriminator(real_data).view(-1,1)
	loss_real = loss_fun(prediction_real,true_target(N,device))
	loss_real.backward()

	# 1.2 Train on Fake Data
	prediction_fake = discriminator(fake_data).view(-1,1)
	loss_fake = loss_fun(prediction_fake,fake_target(N,device))
	loss_fake.backward()
	
	# 1.3 Update weights with gradients
	optimizer.step()
	return loss_real + loss_fake

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	image_size = 28
	compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
	dataset = datasets.MNIST(root='MNIST',train=True,transform=compose,download=True)

	batch_size = 128
	m = int(np.sqrt(batch_size))-1 # the sqrt of number of test samples
	lr = 1e-4
	dropout_prob = 0.3
	noise_dim = 100
	output_size = 1

	num_test_samples = m**2
	test_noise = sample_noise(num_test_samples,noise_dim,device)[:,:,None,None]

	# intialize models
	generator = ConvGenerator(input_size=noise_dim,hidden_size=32,output_size=output_size).to(device)
	discriminator = ConvDiscriminator(input_size=output_size,hidden_size=32,output_size=output_size).to(device)

	# otpimizers
	g_optimizer = optim.Adam(generator.parameters(),lr=lr)
	d_optimizer = optim.Adam(discriminator.parameters(),lr=lr)
	loss_fun = nn.BCELoss()

	# data loader
	data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

	# Create logger instance
	dis = updatable_display2(['train'],["epoch","d_error","g_error"])
	# Total number of epochs to train
	num_epochs = 200
	global_step = 0
	epoch = 0
	d_error = 0
	g_error = 0

	gen_steps = 1
	gen_train_freq = 5
	try:
		while epoch < num_epochs:
			for n_batch,(real_batch,_) in enumerate(data_loader):
				N = real_batch.size(0)
				# 1. Train Discriminator
				real_data = real_batch.to(device)
				# Generate fake data and detach 
				# (so gradients are not calculated for generator)
				noise_tensor = sample_noise(N,noise_dim,device)[:,:,None,None]
				with torch.no_grad():
					fake_data = generator(noise_tensor).detach()
				# Train D
				d_error = train_discriminator(real_data,fake_data,d_optimizer)

				# 2. Train Generator every 'gen_train_freq' steps
				if global_step % gen_train_freq == 0:
					for _ in range(gen_steps):
						# Generate fake data
						fake_data = generator(sample_noise(N,noise_dim,device)[:,:,None,None])
						# Train G
						g_error = train_generator(fake_data,g_optimizer)
						g_error = g_error.item()

				# Log batch error and delete tensors
				dis.update(global_step,'train',{"epoch":epoch,"d_error":d_error.item(),"g_error":g_error} )
				global_step += 1
				del fake_data
				del real_data
				del noise_tensor

				# Display Progress every few batches
				if global_step % 50 == 0:
					test_images = vec2img(generator(test_noise),image_size)
					test_images = test_images.data
					canvas = np.zeros((image_size*m,image_size*m))
					q = 0
					for i in range(m):
						for j in range(m):
							canvas[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size] = test_images[q]
							q += 1
					plt.figure(figsize=(10,10))
					plt.imshow(canvas,cmap='gray')
					plt.axis("off")
					if epoch % 50 == 0:
						dis.save("Figures/DCGAN-MNIST-Epoch="+str(epoch)+".pkl")
						plt.savefig("Figures/DCGAN-MNIST-Epoch="+str(epoch)+".png")
			epoch += 1
	except:
		test_images = vec2img(generator(test_noise),image_size)
		test_images = test_images.data
		canvas = np.zeros((image_size*m,image_size*m))
		q = 0
		for i in range(m):
			for j in range(m):
				canvas[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size] = test_images[q]
				q+=1
		dis.save("Figures/DCGAN-MNIST.pkl")
		plt.figure(figsize=(10,10))
		plt.imshow(canvas,cmap='gray')
		plt.axis("off")
		plt.savefig("Figures/DCGAN-MNIST.png")