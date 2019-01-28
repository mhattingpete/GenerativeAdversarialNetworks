import bcolz
import pickle
import numpy as np

words = []
idx = 0
word2idx = {}
emb_dim = 50
glove_name = f"GloVe/6B.{emb_dim}"
vectors = bcolz.carray(np.zeros(1),rootdir=f"{glove_name}.dat", mode="w")

with open(f"glove.{glove_name}d.txt","rb") as f:
	for l in f:
		line = l.decode().split()
		word = line[0]
		words.append(word)
		word2idx[word] = idx
		idx += 1
		vect = np.array(line[1:]).astype(np.float)
		vectors.append(vect)
	
vectors = bcolz.carray(vectors[1:].reshape((int(vectors[1:].shape[0]/emb_dim),emb_dim)),rootdir=f"{glove_name}.dat",mode="w")
vectors.flush()
pickle.dump(words,open(f"{glove_name}_words.pkl","wb"))
pickle.dump(word2idx,open(f"{glove_name}_idx.pkl","wb"))