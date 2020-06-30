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
import numpy as np
from copy import deepcopy
from torch.utils.data import Subset as StdSubset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from torchvision import transforms

import random
random.seed(params.SEED)

def incrementalTrain(task, trainDS, ICaRL, exemplars, transformer):
	trainSplits = trainDS.splits
	
	train_indexes = trainDS.__getIndexesGroups__(task)


	col = []
	for i,x in enumerate( trainSplits[ :int(task/10) + 1]) : #comprende le 10 classi di questo task
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = col.astype(int)
	print('col = ', col)
	print('col[:10]',  np.array(col[:10]))

	ICaRL = updateRep(task, trainDS, train_indexes, ICaRL, exemplars, trainSplits, transformer)

	m = params.K/(task + params.TASK_SIZE)
	m = int(m+1) #arrotondo per eccesso; preferisco avere max 100 exemplars in più che non 100 in meno
	exemplars = reduceExemplars(exemplars,m)

	exemplars = generateNewExemplars(exemplars, m, col[task:], trainDS, train_indexes, ICaRL)

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
			labels = labels.to(params.DEVICE)
			onehot_labels = torch.eye(100)[labels].to(params.DEVICE)
			mappedLabels = utils.mapFunction(labels, col)
			
			optimizer.zero_grad()
			
			outputs = ICaRL(images, features = False)
			old_outputs = old_ICaRL(images, features = False)
			weights = torch.sum( onehot_labels, dim=0)/torch.sum(onehot_labels) #prova con media fatta sul batch corrente
			#print(weights)
			loss = utils.calculateLoss(outputs, old_outputs, onehot_labels, task, splits)
			
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
	exemplars = copy.deepcopy(exemplars)
	
	for i, el in enumerate(exemplars):
		if el is not None:
			exemplars[i] = el[:m]
	return exemplars	

def generateNewExemplars(exemplars, m, col, trainDS, train_indexes, ICaRL):
	#col contiene i valori delle 10 classi in analisi in questo momento
	exemplars = deepcopy(exemplars)
	for classe in col:
		idxsImages = []
		for i in train_indexes:
			image, label, idx = trainDS.__getitem__(i)
			if( label == classe ):
				idxsImages.append(idx)
		##print('immagini nuova classe ', classe, ' sono: ', len(idxsImages))
		exemplars[classe] = constructExemplars(idxsImages, m, ICaRL, trainDS)
	return exemplars

def constructExemplars(idxsImages, m, ICaRL, trainDS):
	ICaRL = deepcopy(ICaRL).train(False)

	ds = trainDS
	ss = StdSubset(ds, idxsImages)

	loader = DataLoader( ss, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
	features = []
	means = torch.zeros( (1, 64)).to(params.DEVICE)
	for i, (image, label, idx) in enumerate(loader):
		with torch.no_grad():
			image = image.float().to(params.DEVICE)
			x = ICaRL( image, features = True)
			x /= torch.norm(x, p=2)
		for s in x:
			features.append(np.array(s.data.cpu()))
		##print(' features dovrebbe avere dimensione i*batchSize, 64')
		##print('shape = ', len(features), '  ', features[0].size)
		ma = torch.sum(x, dim=0) #sommo sulle colonne, ovvero sulle features
		means += ma
	means = means/ len(idxsImages) # medio
	means = means / means.norm()
	means = means.data.cpu().numpy()
	newExs = []
	phiNewEx = []
	mapFeatures = np.arange( len(features) )
	for k in range (0, m):
		phiX = features #le features di tutte le immagini della classe Y ---> rige = len(ss); colonne = #features
		phiP = np.sum(phiNewEx, axis = 0) #somma su tutte le colonne degli exemplars esistenti. Quindi ogni colonna di phiP sarà la somma del valore di quella feature per ognuna degli exemplars
		mu1 = 1/(k+1)* ( phiX + phiP)
		idxEx = np.argmin(np.sqrt(np.sum((means - mu1) ** 2, axis=1))) #compute the euclidean norm among all the rows in phiX
		newExs.append(idxsImages[mapFeatures[idxEx]])
		phiNewEx.append(features[idxEx])
		features.pop(idxEx)
		mapFeatures = np.delete(mapFeatures, idxEx) 
		
	return newExs

def classify(images, exemplars, ICaRL, task, trainDS):

	ICaRL.train(False)
	images = images.float().to(params.DEVICE)
	phiX = ICaRL(images, features = True)
	phiX /= torch.norm(phiX, p=2)

	trainSet = []
	for classe in exemplars:
		if( classe is not None):
			trainSet = np.concatenate( (trainSet, classe) )
	trainSet = trainSet.astype(int)

	ss = StdSubset(trainDS, trainSet)
	loader = DataLoader( ss, num_workers=params.NUM_WORKERS, batch_size=256)
	X_train = []
	y_train = []
	for i, (image, label, idx) in enumerate(loader):
		with torch.no_grad():
			image = image.float().to(params.DEVICE)
			x = ICaRL( image, features = True)
			x /= torch.norm(x, p=2)
		for s in x:
			X_train.append(np.array(s.data.cpu()))
		y_train = np.concatenate( (y_train, label) )
				
		

	model = KNeighborsClassifier(n_neighbors=3)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)

	#y_train=np.stack( y_train, axis=0 )

	model.fit(X_train, y_train)#Reshaped

	X = []

	for data in phiX:
		data=np.array(data.detach().cpu())
		X.append(data)
		
		
	X = scaler.transform(X)

	preds = model.predict(X)

	return (torch.tensor(preds))
