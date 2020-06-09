from torchvision.datasets import CIFAR100
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from DatasetCIFAR import params
import random
random.seed(42)

class Dataset(CIFAR100):

	'''
	The class Dataset define methods and attributes to manage a CIFAR100 dataset
	Attributes:
		train = Bool, default value = True
		Transform
		Target_Transform

		_dataset = contains the pythorch dataset CIFAR100 defined by hyperparameters passes by input
		_targets = contains a lisf of 60000 elements, and each one referred to an image of the dataset. The value of each element is an integer in [0,99] that explicit the label for that image
		            E.g. _targets[100] get the label of the 100th image in the dataset
		_data = contains a list of 60000 imagest, each one represented by a [32]x[32]x[3] vector that define pixels 
		_labelNames = contains a list of 100 elements, each one represent a class; it maps integer indexes to human readable labels
	'''
	def __getClassesNames__(self):
		#This method returns a list mapping the 100 classes into a human readable label. E.g. names[0] is the label that maps the class 0
		names = []
		classi = list(self._dataset.class_to_idx.keys())
		for i in self.searched_classes:
		  names.append(classi[int(i)])
		self._labelNames = names
		return names

	def __init__(self, root = 'Dataset', classes=range(10), train=True, transform=None, target_transform=None, download=True):
		
		super(Dataset, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

		# Select subset of classes
		
		data = []
		targets = []

		for i in range(len(self.data)):
		        data.append(self.data[i])
		        targets.append(self.targets[i])

		self._data = np.array(data)
		self._targets = np.array(targets) #Essendo CIFAR100 una sottoclasse di CIFAR10, qui fa riferimento a quell'implementazione.
		self.splits = params.returnSplits()



	def __getIndexesGroups__(self, index = 0):
		#This method returns a list containing the indexes of all the images belonging to classes [starIndex, startIndex + 10]
		indexes = []
		self.searched_classes = self.splits[int(index/10)]
		i = 0
		for el in self._targets:
		  if (el in self.searched_classes):
		    indexes.append(i)
		  i+=1
		return indexes
	
	def __getitem__(self, idx):
		#Given an index, this method return the image and the class corresponding to that index
		image = np.transpose(self._data[idx])
		label = self._targets[idx]
		return image, label

	def __len__(self):
		return len(self._targets)
