from DatasetCIFAR.data_set import Dataset 
from DatasetCIFAR.data_set import Subset 
from DatasetCIFAR import ResNet
from DatasetCIFAR import utils
from DatasetCIFAR import params
from DatasetCIFAR import ICaRLModel
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as f
import copy
import sys
import numpy as np
from copy import deepcopy
from torch.utils.data import Subset as StdSubset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
import random
random.seed(params.SEED)

def incrementalTrain(task, trainDS, ICaRL, exemplars, transformer, randomS = False):
	trainSplits = trainDS.splits
	
	train_indexes = trainDS.__getIndexesGroups__(task)

	col = []
	for i,x in enumerate( trainSplits[ :int(task/10) + 1]) : #comprende le 10 classi di questo task
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = col.astype(int)

	ICaRL = updateRep(task, trainDS, train_indexes, ICaRL, exemplars, trainSplits, transformer)

	m = params.K/(task + params.TASK_SIZE)
	m = int(m + .5) #arrotondo per eccesso; preferisco avere max 100 exemplars in più che non 100 in meno
	exemplars = reduceExemplars(exemplars,m)

	exemplars = generateNewExemplars(exemplars, m, col[task:], trainDS, train_indexes, ICaRL, randomS = randomS)

	return ICaRL, exemplars

def updateRep(task, trainDS, train_indexes, ICaRL, exemplars, splits, transformer):

	dataIdx = np.array(train_indexes)
	for classe in exemplars:
		if( classe is not None):
			#print('classe = ', classe)
			dataIdx = np.concatenate( (dataIdx, classe) )

	#dataIdx contiene gli indici delle immagini, in train DS, delle nuove classi e dei vecchi exemplars

	D = Subset(trainDS, dataIdx, transformer)

	loader = DataLoader( D, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle = True)

	old_ICaRL = deepcopy(ICaRL)

	old_ICaRL.train(False)
	ICaRL.train(True)

	optimizer = torch.optim.SGD(ICaRL.parameters(), lr=params.LR, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA) #allow to change the LR at predefined epochs
	current_step = 0
	col = []
	for i,x in enumerate( splits[ :int(task/10) + 1]):
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = np.array(col).astype(int)

	for epoch in range(params.NUM_EPOCHS):
		lenght = 0
		running_corrects = 0
  	  

		for images, labels, idx in loader:
			images = images.float().to(params.DEVICE)
			labels = labels.long().to(params.DEVICE)
			
			#onehot_labels = torch.eye(100)[labels].to(params.DEVICE)
			
			mappedLabels = utils.mapFunction(labels, col)
			
			optimizer.zero_grad()
			
			outputs = ICaRL(images, features = False)
			old_outputs = old_ICaRL(images, features = False)
			#weights = torch.sum( onehot_labels, dim=0)/torch.sum(onehot_labels) #prova con media fatta sul batch corrente
			weights = utils.generateWeights(task, col).to(params.DEVICE)

			loss = utils.calculateLoss(outputs, old_outputs, labels, task, splits, typeLoss = 'LogLoss', weights = weights)
			
			cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), col[None,:], axis = 1).to(params.DEVICE)
			_ , preds = torch.max(cut_outputs.data, 1)
			
			running_corrects += torch.sum(preds == mappedLabels.data).data.item()
			lenght += len(images)
			
			loss.backward()  # backward pass: computes gradients
			optimizer.step()
		accuracy = running_corrects / float(lenght)
		scheduler.step()
		print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ", accuracy)
	return ICaRL

def reduceExemplars(exemplars,m):
	exemplars = copy.deepcopy(exemplars)
	
	for i, el in enumerate(exemplars):
		if el is not None:
			exemplars[i] = el[:m]
	return exemplars	

def generateNewExemplars(exemplars, m, col, trainDS, train_indexes, ICaRL, randomS = False):
	#col contiene i valori delle 10 classi in analisi in questo momento
	exemplars = deepcopy(exemplars)
	for classe in col:
		idxsImages = []
		for i in train_indexes:
			image, label, idx = trainDS.__getitem__(i)
			if( label == classe ):
				idxsImages.append(idx)
		##print('immagini nuova classe ', classe, ' sono: ', len(idxsImages))
		if(randomS is not True):
			exemplars[classe] = constructExemplars(idxsImages, m, ICaRL, trainDS)
		else:
			exemplars[classe] = random.sample(idxsImages, m)
	return exemplars

def constructExemplars(idxsImages, m, ICaRL, trainDS):
	ICaRL = ICaRL.train(False)

	features = []
	with torch.no_grad():
		for idx in idxsImages:
			img, lbl, _ = trainDS.__getitem__(idx)
			img = torch.tensor(img).unsqueeze(0).float()
			x = ICaRL( img.to(params.DEVICE) , features = True).data.cpu().numpy()
			x = x / np.linalg.norm(x) 
			features.append(x[0])
	features = np.array(features)
	means = np.mean(features, axis=0)
	means = means / np.linalg.norm(means)

	newExs = []
	phiNewEx = []
	idxsImages = deepcopy(idxsImages)
	for k in range (0, m):
		phiX = features #le features di tutte le immagini della classe Y ---> rige = len(ss); colonne = #features
		phiP = np.sum(phiNewEx, axis = 0) #somma su tutte le colonne degli exemplars esistenti. Quindi ogni colonna di phiP sarà la somma del valore di quella feature per ognuna degli exemplars
		mu1 = 1/(k+1)* ( phiX + phiP)
		idxEx = np.argmin(np.sqrt(np.sum((means - mu1) ** 2, axis=1))) #compute the euclidean norm among all the rows in phiX
		newExs.append(idxsImages[idxEx])
		phiNewEx.append(features[idxEx])
		features = np.delete(features, idxEx, axis = 0)
		idxsImages.pop(idxEx)
	#print('Selected: ',newExs)
	return newExs

def classify(images, exemplars, ICaRL, task, trainDS, mean = None):
	preds = []

	nClasses = task + params.TASK_SIZE
	means = torch.zeros( ( nClasses, 64)).to(params.DEVICE)

	ICaRL.train(False)
	images = images.float().to(params.DEVICE)
	phiX = ICaRL(images, features = True)
	phiX /= torch.norm(phiX, p=2)

	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	ds = trainDS
	classiAnalizzate = []
	if(mean == None):
		for i in range( 0, int(task/10) + 1) :
			##print('split i = ', ds.splits[i])
			classiAnalizzate = np.concatenate( (classiAnalizzate, ds.splits[i]) )
		##print('classi = ', classiAnalizzate)
		for y in range (0, task + params.TASK_SIZE):
			#now idxsImages contains the list of all the images selected as exemplars
			classY = int(classiAnalizzate[y])
			ss = Subset(ds, exemplars[classY], transformer)
			loader = DataLoader( ss, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
			for img, lbl, idx in loader:
				with torch.no_grad():
					img = img.float().to(params.DEVICE)
					x = ICaRL(img, features = True)
					x /= torch.norm(x, p=2)
				ma = torch.sum(x, dim=0)
				means[y] += ma
			means[y] = means[y]/ len(idx) # medio
			means[y] = means[y] / means[y].norm()

	else:
		means = mean
	for data in phiX:
		#print('shape data = ', data.shape)
		pred = np.argmin(np.sqrt( np.sum((data.data.cpu().numpy() - means.data.cpu().numpy())**2, axis = 1 )   ) )
		
		preds.append(pred)

	return (torch.tensor(preds), means)
