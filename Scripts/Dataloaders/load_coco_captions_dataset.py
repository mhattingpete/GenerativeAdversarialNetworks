from Dataloaders.cococaptions import CocoCaptions
from torchtext.data import Field,BucketIterator
from torch.utils.data import DataLoader,random_split
import os
import re

def create_dataset(batch_size,device,split_ratio=0.7,min_vocab_freq=10,max_vocab_size=4000):
	text_field = Field(tokenize="spacy",tokenizer_language="en",batch_first=True,init_token="<sos>",eos_token="<eos>",lower=True)

	path_to_dataset = "/home/martin/Documents/Datasets/COCO2015Captions"

	def transform(caption):
		caption = caption.strip().lower().split()
		return caption

	dataset = CocoCaptions(annFile=os.path.join(path_to_dataset,"captions_train2014.json"),text_field=text_field,transform=transform)
	train,val = dataset.split(split_ratio=split_ratio)
	test = CocoCaptions(annFile=os.path.join(path_to_dataset,"captions_val2014.json"),text_field=text_field,transform=transform)

	print("Dataset loaded")

	text_field.build_vocab(dataset.text,min_freq=min_vocab_freq,max_size=max_vocab_size)
	SOS_TOKEN = text_field.vocab.stoi['<sos>']
	EOS_TOKEN = text_field.vocab.stoi['<eos>']
	UNK_TOKEN = text_field.vocab.stoi['<unk>']
	PAD_TOKEN = text_field.vocab.stoi['<pad>']

	print("Vocabuly build")

	print("Vocabuly statistics")

	print("\nMost common words in the vocabulary:\n",text_field.vocab.freqs.most_common(10))
	print("Size of the vocabulary:",len(text_field.vocab))
	print("Max sequence lenght",dataset.max_seq_len)

	train_iter,val_iter = BucketIterator.splits((train,val),repeat=True,batch_size=batch_size,device=device)
	test_iter = BucketIterator(test,batch_size=batch_size,repeat=True,train=False,device=device)
	return {"data_iters":(train_iter,val_iter,test_iter),"fields":text_field,"num_classes":len(text_field.vocab),
	"tokens":(SOS_TOKEN,EOS_TOKEN,UNK_TOKEN,PAD_TOKEN),"max_seq_len":dataset.max_seq_len}

"""
print("Start iterating through data")

for i,batch in enumerate(train_iter):
	print(batch.text)
	break

for i,batch in enumerate(val_iter):
	print(batch.text)
	break

for i,batch in enumerate(test_iter):
	print(batch.text)
	break
"""