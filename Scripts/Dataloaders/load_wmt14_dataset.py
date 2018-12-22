import torchtext.datasets as datasets
from torchtext.data import Field,BucketIterator
import os
from torch.utils.data import DataLoader
import re
import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')

def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@',text))]

def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@',text))]

data_path = "/home/martin/Documents/Datasets"

#EN = Field(tokenize=tokenize_en,batch_first=True,init_token="<SOS>",eos_token="<EOS>")
#DE = Field(tokenize=tokenize_de,batch_first=True,init_token="<SOS>",eos_token="<EOS>")
EN = Field(tokenize="spacy",tokenizer_language="en",batch_first=True,init_token="<SOS>",eos_token="<EOS>")
DE = Field(tokenize="spacy",tokenizer_language="de",batch_first=True,init_token="<SOS>",eos_token="<EOS>")

# multi30k dataloader
train,val,test = datasets.Multi30k.splits(exts=(".en",".de"),fields=(EN,DE),root=data_path)

# wmt14 dataloader (better than using datasets.WMT14.splits since it's slow)
#train,val,test = datasets.TranslationDataset.splits(exts=(".en",".de"),fields=[("src",EN),("trg",DE)],path=os.path.join(data_path,"wmt14"),
#	train="train.tok.clean.bpe.32000",validation="newstest2013.tok.bpe.32000",test="newstest2014.tok.bpe.32000")

print("Dataset loaded")

EN.build_vocab(train.src,min_freq=3)
DE.build_vocab(train.trg,max_size=50000)

print("Vocabularies build")

train_iter,val_iter = BucketIterator.splits((train, val),batch_size=3)
test_iter = BucketIterator(test,batch_size=3)

print("Start iterating through data")

for i,batch in enumerate(train_iter):
	print(batch.src) # the source language
	print(batch.trg) # the target language
	break

for i,batch in enumerate(val_iter):
	print(batch.src) # the source language
	print(batch.trg) # the target language
	break

for i,batch in enumerate(test_iter):
	print(batch.src) # the source language
	print(batch.trg) # the target language
	break

print("Vocabularies statistics")

print("\nMost common words in the english vocabulary:\n",EN.vocab.freqs.most_common(10))
print("Size of the english vocabulary:",len(EN.vocab))
print("\nMost common words in the german vocabulary:\n",DE.vocab.freqs.most_common(10))
print("Size of the german vocabulary:",len(DE.vocab))