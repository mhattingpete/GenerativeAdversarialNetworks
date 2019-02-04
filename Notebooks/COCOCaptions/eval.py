import sys
sys.path.append("../../Scripts")

import torch
import os
import json

with open(sys.argv[1],"rb") as f:
	config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

import numpy as np
from torch import nn
import random
from nltk.translate.bleu_score import sentence_bleu

from utils import num_parameters,sample_noise,load_model,tensor_to_list_of_words
import generators

from Dataloaders.load_coco_captions_dataset import create_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = os.path.join("../Saved_models",config["model_config"]["save_name"])

summary_path = os.path.join(model_save_path,"train/exp-0")

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
noise_size = config["model_config"]["noise_size"]

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

load_model(generator,summary_path)

text_log = open(os.path.join(summary_path,"eval_log.txt"),"a")

# losses
loss_weight = torch.ones(num_classes).to(device)
loss_weight[SOS_TOKEN] = 0.0
#loss_weight[EOS_TOKEN] = 0.0
#loss_weight[UNK_TOKEN] = 0.0
loss_weight[PAD_TOKEN] = 0.0
pretrain_loss_fun = nn.NLLLoss(weight=loss_weight)

np_g = num_parameters(generator)
text_log.write("Number of parameters for G: {}\n"
	  .format(np_g))

max_temperature = torch.FloatTensor([config["train_config"]["max_temperature"]]).to(device)

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