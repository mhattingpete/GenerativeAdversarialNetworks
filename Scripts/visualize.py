import matplotlib.pyplot as plt
import numpy as np

def plotSamples(x,xu=None,yu=1,max_print=100,title="Data"):
	x = x.cpu().data.numpy()
	fig = plt.figure()
	fig.suptitle(title)
	n = int(np.floor(np.sqrt(max_print)))
	nsq = n*n
	if xu == None:
		xu = x.shape[1]
	for i in range(x.shape[0]):
		if i >= nsq:
			break
		ax = fig.add_subplot(n,n,i+1)
		ax.plot(x[i,:],'r')
		ax.set_xlim([0,xu])
		ax.set_ylim([0,yu])
		ax.axis('off')

def tensor_to_words(batch,num_to_word_vocab):
	text_translated = []
	for line in batch:
		line_translated = []
		for word in line:
			word_tranlated = num_to_word_vocab[word.cpu().numpy().tolist()]
			if word_tranlated in ["<sos>","<eos>","<pad>"]:
				continue
			line_translated.append(word_tranlated)
		text_translated.append(" ".join(line_translated))
	return "\n".join(text_translated)
