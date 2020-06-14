from DatasetCIFAR.data_set import Dataset 
from DatasetCIFAR import ResNet
from DatasetCIFAR import utils
from DatasetCIFAR import params
from DatasetCIFAR import ICaRLModel
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import numpy as np

from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torch.nn import functional as F
import random
random.seed(params.SEED)

def incrementalTrain(task, trainDS, ICaRL, exemplars):
	trainSplits = trainDS.splits
	
	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_indexes = trainDS.__getIndexesGroups__(task)


	col = []
	for i,x in enumerate( trainSplits[ :int(task/10) + 1]) : #comprende le 10 classi di questo task
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = col.astype(int)
	print('col = ', col)
	print('col[:10]',  col[:10])

	train_ds = Dataset(train=True, transform = transformer)
	train_ds._data = trainDS._data[train_indexes]
	train_ds._targets = trainDS._targets[train_indexes]
	
	current_exemplars = exemplars[col[:10]]
	ICaRL = updateRep(task, train_ds, ICaRL, exemplars, trainSplits)

	m = params.K/(task + params.TASK_SIZE)

	exemplars = reduceExemplars(exemplars,m)

	exemplars = generateNewExemplars(exemplars, m, col[:10], trainDS, train_indexes, ICaRL)

	return ICaRL, exemplars

def updateRep(task, train_indexes, ICaRL, exemplars, splits):

	dataIdx = np.array(train_indexes)
	for classe in exemplars:
		if( classe is not None):
			dataIdx = np.concatenate(dataIdx, classe)

	#dataIdx contiene gli indici delle immagini, in train DS, delle nuove classi e dei vecchi exemplars

	D = Subset(trainDS, dataIdx)

	loader = DataLoader( D, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle = True)

	old_ICaRL = deepcopy(ICaRL)

	old_ICaRL.train(False)
	ICaRL.train(True)

	optimizer = torch.optim.SGD(ICaRL.parameters(), lr=params.LR, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA) #allow to change the LR at predefined epochs
	current_step = 0

	for i,x in enumerate( splits[ :int(task/10) + 1]):
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = np.array(col).astype(int)

	for epoch in range(params.NUM_EPOCHS):
		lenght = 0
		running_corrects = 0
  	  

	for images, labels, idx in loader:
		images = images.float().to(params.DEVICE)
		labels = labels.to(params.DEVICE)
		onehot_labels = torch.eye(100)[labels].to(params.DEVICE)
		mappedLabels = utils.mapFunction(labels, col)
		
		optimizer.zero_grad()
		
		outputs = ICaRL(images, features = False)
		old_outputs = old_ICaRL(images, features = False)
		
		loss = utils.calculateLoss(outputs, old_outputs, onehot_labels, task, splits )
		
		cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), col[None,:], axis = 1).to(params.DEVICE)
		_ , preds = torch.max(cut_outputs.data, 1)
		
		running_corrects += torch.sum(preds == mappedLabels.data).data.item()
		lenght += len(images)
		
		loss.backward()  # backward pass: computes gradients
		optimizer.step()
	accuracy = running_corrects / float(lenght)
	print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ", accuracy)
	
	return ICaRL

def reduceExemplars(exemplars,m):
	exemplars = exemplars.copy
	for i, el in enumerate(exemplars):
		if el is not None:
			exemplars[i] = el[:m,]
	return exemplars	

def generateNewExemplars(exemplars, m, col, trainDS, train_indexes, ICaRL):
	#col contiene i valori delle 10 classi in analisi in questo momento
	exemplars = exemplars.copy
	for classe in col:

		idxsImages = []
		for i in train_indexes:
			image, label, idx = trainDS.__getitem__(i)
			if( label == classe ):
				idxsImages.append(idx)
		exemplars[classe] = constructExemplars(idxsImages, m, ICaRL)
	return exemplars

def constructExemplars(idxsImages, m, ICaRL):
	ICaRL = deepcopy(ICaRL).train(False)
	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

	ds = Dataset(train = True, transform = transformer)

	ss = Subset(ds, idxsImages)

	loader = DataLoader( ss, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
	features = []
	means = torch.zeros( (1, 64)).to(params.DEVICE)
	for i, (image, label, idx) in enumerate(loader):
		with torch.no_grad():
			x = ICaRL( image, features = True)
		for s in x:
			features.append(np.array(s.data.cpu()))
		ma = torch.sum(x, dim=0) #sommo sulle colonne, ovvero sulle features
		means += ma
	means = means/ len(idxsImages) # medio

	newExs = []
	phiNewEx = []
	for k in range (0, m):
		phiX = features #le features di tutte le immagini della classe Y
		phiP = np.sum(phiNewEx, axis = 0) #ad ogni step k, ho già collezionato k-1 examplars
		mu1 = 1/(k+1)* ( phiX + phiP)
		idxEx = np.argmin(np.sqrt(np.sum((mu - mu1) ** 2, axis=1))) #compute the euclidean norm among all the rows in phiX
		newExs.append(idxsImages[idxEx])
		phiExemplaresY.append(features[idxEx])
	return newExs

def classify(images, exemplars, ICaRL, task):
	preds = []

	nClasses = task + params.TASK_SIZE
	means = torch.zeros( ( nClasses, 64)).to(params.DEVICE)

	ICaRL.train(False)

	phiX = ICaRL(images, features = True)

	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	ds = Dataset(train=True, transform = transformer)
	classiAnalizzate = []

	for i in range( 0, int(task/10) + 1) :
		classiAnalizzate.append(ds.trainSplits[i])

	for y in range (0, task + params.TASK_SIZE):
		#now idxsImages contains the list of all the images selected as exemplars
		classY = classiAnalizzate[y]
		ss = Subset(ds, exemplars[classY])
		loader = DataLoader( ss, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
		for img, lbl, idx in loader:
			with torch.no_grad:
				x = ICaRL(img, features = True)
			ma = torch.sum(x, dim=0)
			means[y] += ma

		means[y] = means[y]/ len(idx) # medio

	for data in phiX:
		pred = np.argmin(np.sqrt( np.sum((phiX - means)**2, axis=1)  ) )
		preds.append(pred)

	return torch.tensor(preds)
