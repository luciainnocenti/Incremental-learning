from DatasetCIFAR.data_set import Dataset 
from DatasetCIFAR import ResNet
from DatasetCIFAR import params
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torch.nn import functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def generateWeights(task, col):
	i = 0
	w = task + 10
	weights = torch.zeros(100)
	for el in col:
		weights[el] = w / params.TASK_SIZE
		if( i == 9):
			w -= 10
			i = 0
		else:
			i+= 1
	return weights

def mapFunction(labels, splits):
	m_l = []
	l_splits = list(splits)
	for el in labels:
		m_l.append( l_splits.index(el) )
	return torch.LongTensor(m_l).to(params.DEVICE)

def trainfunction(task, train_loader, train_splits):
	print(f'task = {task} ')
	resNet = torch.load('resNet_task' + str(task) + '.pt').train(True)
	old_resNet = torch.load('resNet_task' + str(task) + '.pt').train(False)

	#Define the parameters for traininig:
	optimizer = torch.optim.SGD(resNet.parameters(), lr=params.LR, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA) #allow to change the LR at predefined epochs
	current_step = 0
	
	col = np.array(train_splits[int(task/10)]).astype(int)
	print("train col = ", col)
	print("train col = ", col[None, :])
	##Train phase
	for epoch in range(params.NUM_EPOCHS):
		lenght = 0
		scheduler.step() #update the learning rate
		running_corrects = 0

		for images, labels, _ in train_loader:
			images = images.float().to(params.DEVICE)
			labels = labels.to(params.DEVICE)
			#print(labels)
			mappedLabels = mapFunction(labels, col)
			#print(mappedLabels)
			
			#onehot_labels = torch.eye(100)[labels].to(params.DEVICE)#it creates the one-hot-encoding list for the labels; needed for BCELoss
			
			optimizer.zero_grad() # Zero-ing the gradients
			
			# Forward pass to the network
			old_outputs = old_resNet(images)
			outputs = resNet(images)
			loss = calculateLoss(outputs, old_outputs, labels, task, train_splits )
			
			
			# Get predictions		
			
			cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), col[None, :], axis = 1).to(params.DEVICE)
			_, preds = torch.max(cut_outputs.data, 1)
			#print(preds)
			
			# Update Corrects
			running_corrects += torch.sum(preds == mappedLabels.data).data.item()
			loss.backward()  # backward pass: computes gradients
			optimizer.step() # update weights based on accumulated gradients
			
			current_step += 1
			lenght += len(images)
		# Calculate Accuracy
		accuracy = running_corrects / float(lenght)
		print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ", accuracy)
	torch.save(resNet, 'resNet_task{0}.pt'.format(task + 10))


def evaluationTest(task, test_loader, test_splits):
	criterion = torch.nn.BCEWithLogitsLoss()
	t_l = 0
	resNet = torch.load('resNet_task' + str(task + 10) + '.pt').eval()# Set Network to evaluation mode
	running_corrects = 0
	
	col = []
	#in fase di test verifico su tutti le classi viste fino ad ora, quindi prendo da test splits gli indici dei gruppi da 0 a task
	for i,x in enumerate( test_splits[ :int(task/10) + 1]):
		 v = np.array(x)
		 col = np.concatenate( (col,v), axis = None)
	col = col.astype(int)
	tot_preds = []
	tot_lab = []
	for images, labels, _ in test_loader:
		images = images.float().to(params.DEVICE)
		labels = labels.to(params.DEVICE)
		mappedLabels = mapFunction(labels, col)
		#M1 onehot_labels = torch.eye(task + params.TASK_SIZE)[mappedLabels].to(params.DEVICE) #it creates the one-hot-encoding list for the labels; neede for BCELoss
		onehot_labels = torch.eye(100)[labels].to(params.DEVICE)
		# Forward Pass
		outputs = resNet(images)
		# Get predictions
		outputs = outputs.to(params.DEVICE)
		
		cut_outputs = np.take_along_axis(outputs, col[None, :], axis = 1)
		cut_outputs = cut_outputs.to(params.DEVICE)
		_, preds = torch.max(cut_outputs.data, 1)
		tot_preds = np.concatenate( ( tot_preds, preds.data.cpu().numpy() ) )
		tot_lab = np.concatenate( (tot_lab, mappedLabels.data.cpu().numpy()  ) )
		# Update Corrects
		running_corrects += torch.sum(preds == mappedLabels.data).data.item()
		print(len(images))
		t_l += len(images)
	# Calculate Accuracy
	accuracy = running_corrects / float(t_l)
	
	#Calculate Loss
	
	loss = criterion(outputs,onehot_labels)
	print('Test Loss: {} Test Accuracy : {}'.format(loss.item(),accuracy) )
	cf = confusion_matrix(tot_lab, tot_preds)
	df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))
	sn.set(font_scale=.4) # for label size
	sn.heatmap(df_cm, annot=False)
	plt.show()
	return(accuracy, loss.item())	  

def calculateLoss(outputs, old_outputs, labels, task, train_splits, typeLoss = 'BCE', weights = None):
	#import matplotlib.pyplot as plt
	#plt.matshow(labels[train_splits[ int(task/10) + 1 ] ].cpu().numpy() )
	
	outputs, old_outputs, labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), labels.to(params.DEVICE)
	col = []
	for i,x in enumerate( train_splits[ :int(task/10) ]):
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = np.array(col).astype(int)
	
	classCriterion = nn.CrossEntropyLoss()
	distCriterion = nn.CosineEmbeddingLoss()
	
	if( task == 0):
		loss = classCriterion(outputs, labels)
		
	if( task > 0 ):
		ys = torch.ones(len(labels)).to(params.DEVICE)
		classLoss = classCriterion(outputs, labels)
		distLoss = distCriterion(outputs[:, col], old_outputs[:, col], ys )
		loss = classLoss + distLoss
	return loss
