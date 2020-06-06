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
import matplotlib.pyplot as plt

def mapFunction(labels, splits):
	m_l = []
	l_splits = list(splits)
	for el in labels:
		m_l.append( l_splits.index(el) )
	return torch.LongTensor(m_l).to(params.DEVICE)

def trainfunction(task, train_loader, train_splits, test_loader, test_splits):
	print(f'task = {task} ')
	resNet = torch.load('resNet_task' + str(task) + '.pt')
	old_resNet = torch.load('resNet_task' + str(task) + '.pt')

	#Define the parameters for traininig:
	optimizer = torch.optim.SGD(resNet.parameters(), lr=2.)
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
		resNet.train(True)
		for images, labels in train_loader:
			images = images.float().to(params.DEVICE)
			labels = labels.to(params.DEVICE)
			#print(labels)
			mappedLabels = mapFunction(labels, col)
			#print(mappedLabels)
			onehot_labels = torch.eye(100)[labels].to(params.DEVICE)#it creates the one-hot-encoding list for the labels; needed for BCELoss
			
			optimizer.zero_grad() # Zero-ing the gradients
			
			# Forward pass to the network
			old_outputs = old_resNet(images)
			outputs = resNet(images)
			loss = calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits )
			
			
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


	resNet.eval() # Set Network to evaluation mode
	running_correctsVal = 0
	
	colVal = []
	#in fase di test verifico su tutti le classi viste fino ad ora, quindi prendo da test splits gli indici dei gruppi da 0 a task
	for i,x in enumerate( test_splits[ :int(task/10) + 1]):
		 v = np.array(x)
		 colVal = np.concatenate( (colVal,v), axis = None)
	colVal = colVal.astype(int)
	
	for imagesVal, labelsVal in test_loader:
		imagesVal = imagesVal.float().to(params.DEVICE)
		labelsVal = labelsVal.to(params.DEVICE)
		mappedLabelsVal = mapFunction(labelsVal, colVal)
		onehot_labelsVal = torch.eye(task + params.TASK_SIZE)[mappedLabelsVal].to(params.DEVICE) #it creates the one-hot-encoding list for the labels; neede for BCELoss
		# Forward Pass
		outputsVal = resNet(imagesVal)
		# Get predictions
		outputsVal = outputsVal.to(params.DEVICE)
		
		cut_outputsVal = np.take_along_axis(outputsVal, colVal[None, :], axis = 1)
		cut_outputsVal = cut_outputsVal.to(params.DEVICE)
		_, predsVal = torch.max(cut_outputsVal.data, 1)
		# Update Corrects
		running_correctsVal += torch.sum(predsVal == mappedLabelsVal.data).data.item()
		t_l += len(imagesVal)
	# Calculate Accuracy
	accuracyVal = running_correctsVal / float(t_l)
	#Calculate Loss
	lossVal = F.binary_cross_entropy_with_logits(cut_outputsVal,onehot_labelsVal)

	print('Validation Loss: {} Validation Accuracy : {}'.format(lossVal.item(),accuracyVal) )
	resNet.train(True)
	torch.save(resNet, 'resNet_task{0}.pt'.format(task + 10))  


def calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits):
	m = nn.Sigmoid()
	
	outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), onehot_labels.to(params.DEVICE)

	col = []
	for i,x in enumerate( train_splits[ :int(task/10) ]):
		v = np.array(x)
		col = np.concatenate( (col,v), axis = None)
	col = np.array(col).astype(int)
	
	if( task == 0):
		loss = F.binary_cross_entropy_with_logits(outputs,onehot_labels)
		
	if( task > 0 ):
		target = onehot_labels.clone().to(params.DEVICE)
		target[:, col] = m(old_outputs[:,col]).to(params.DEVICE)
		loss = F.binary_cross_entropy_with_logits( input=outputs, target=target )
	return loss
