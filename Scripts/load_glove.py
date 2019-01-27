import bcolz
import pickle

def load_weight_matrix(target_vocab,emb_dim=50):
	glove_name = f"6B.{emb_dim}"
	glove_path = "GloVe"
	vectors = bcolz.open(f"{glove_path}/{glove_name}.dat")[:]
	words = pickle.load(open(f"{glove_path}/{glove_name}_words.pkl","rb"))
	word2idx = pickle.load(open(f"{glove_path}/{glove_name}_idx.pkl","rb"))
	glove = {w: vectors[word2idx[w]] for w in words}
	matrix_len = len(target_vocab)
	weights_matrix = np.zeros((matrix_len,emb_dim))
	words_found = 0
	for i,word in enumerate(target_vocab):
		try:
			weights_matrix[i] = glove[word]
			words_found += 1
		except KeyError:
			weights_matrix[i] = np.random.normal(scale=0.6,size=(emb_dim,))
	return weights_matrix

def create_emb_layer(target_vocab,emb_dim=50,non_trainable=False):
	assert emb_dim in [50,100,200,300]
	weights_matrix = load_weight_matrix(emb_dim=emb_dim)
	num_embeddings,embedding_dim = weights_matrix.size()
	emb_layer = nn.Embedding(num_embeddings,embedding_dim)
	emb_layer.load_state_dict({"weight":weights_matrix})
	if non_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer,num_embeddings,embedding_dim