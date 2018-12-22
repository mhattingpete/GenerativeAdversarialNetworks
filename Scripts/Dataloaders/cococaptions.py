#from torch.utils.data import Dataset
from torchtext.data import Example,Dataset

import random
from contextlib import contextmanager
from copy import deepcopy

class CocoCaptions(Dataset):
	"""`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
	This is a modification of the dataset from 'https://pytorch.org/docs/stable/_modules/torchvision/datasets/coco.html'
	where this only loads the captions and not the images.

	Args:
		annFile (string): Path to json annotation file.
		transform (callable, optional): A function/transform that takes in the
			caption and transforms it.

	Example:

		.. code:: python

			import torchvision.datasets as dset
			import torchvision.transforms as transforms
			cap = dset.CocoCaptions(annFile = 'json annotation file',
									transform=None)

			print('Number of samples: ', len(cap))
			caption = cap[3] # load 4th sample

			print(caption)

		Output: ::

			Number of samples: 82783
			[u'A plane emitting smoke stream flying over a mountain.',
			u'A plane darts across a bright blue sky behind a mountain covered in snow',
			u'A plane leaves a contrail above the snowy mountain top.',
			u'A mountain that has a plane flying overheard in the distance.',
			u'A mountain view with a plume of smoke in the background']

	"""
	def __init__(self,annFile,text_field,transform=None):
		from pycocotools.coco import COCO
		coco = COCO(annFile)
		ids = list(coco.imgs.keys())
		transform = transform
		field = [("text",text_field)]
		examples = []
		max_seq_len = 0
		for i in ids:
			ann_ids = coco.getAnnIds(imgIds=i)
			anns = coco.loadAnns(ann_ids)
			for ann in anns:
				caption = ann['caption']
				if transform is not None:
					caption = transform(caption)
				if len(caption) > max_seq_len:
					max_seq_len = len(caption)
				examples.append(Example.fromlist([caption],field))
		self.max_seq_len = max_seq_len
		super().__init__(examples=examples,fields=field)

	@staticmethod
	def sort_key(ex):
		return len(ex.text)

	def split(self,split_ratio=0.7):
		"""Create train-valid splits from the instance's examples.
		Arguments:
			split_ratio (float or List of floats): a number [0, 1] denoting the amount
				of data to be used for the training split (rest is used for validation),
				or a list of numbers denoting the relative sizes of train, test and valid
				splits respectively. If the relative size for valid is missing, only the
				train-test split is returned. Default is 0.7 (for the train set).
		Returns:
			Tuple[Dataset]: Datasets for train and validation splits in that order.
		"""
		train_ratio = check_split_ratio(split_ratio)

		# For the permutations
		rnd = RandomShuffler()
		train_data,val_data = rationed_split(self.examples,train_ratio,rnd)
		splits = tuple(Dataset(d,self.fields) for d in (train_data,val_data) if d)

		# In case the parent sort key isn't none
		if self.sort_key:
			for subset in splits:
				subset.sort_key = self.sort_key
		return splits

def check_split_ratio(split_ratio):
	"""Check that the split ratio argument is not malformed"""
	# Assert in bounds, validation size is zero
	assert 0. < split_ratio < 1., ("Split ratio {} not between 0 and 1".format(split_ratio))
	return split_ratio

def rationed_split(examples,train_ratio,rnd):
	# Create a random permutation of examples, then split them
	# by ratio x length slices for each of the train/dev splits
	N = len(examples)
	randperm = rnd(range(N))
	train_len = int(round(train_ratio * N))
	# Due to possible rounding problems
	val_len = N - train_len
	indices = (randperm[:train_len],  # Train
			   randperm[train_len:])  # Validation
	data = tuple([examples[i] for i in index] for index in indices)
	return data

class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""
    def __init__(self,random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
        	return random.sample(data, len(data))