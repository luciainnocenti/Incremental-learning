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
	lab = labels.clone()
	for i, x in enumerate(splits):
		lab[lab == x] = i
	return lab

def trainfunction(task, train_loader, train_splits):
	pars_epoch = [] #clean the pars_epoch after visualizations
	print(f'task = {task} ')
	resNet = torch.load('resNet_task' + str(task) + '.pt')
	old_resNet = torch.load('resNet_task' + str(task) + '.pt')

	#Define the parameters for traininig:
	optimizer = torch.optim.SGD(resNet.parameters(), lr=2.)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA) #allow to change the LR at predefined epochs
	current_step = 0
	
	col = np.array(train_splits[int(task/10)]).astype(int)
	
	##Train phase
	for epoch in range(params.NUM_EPOCHS):
		lenght = 0
		scheduler.step() #update the learning rate
		#print(scheduler.get_lr(), "   ", scheduler.get_last_lr()) #check if the lr is okay
		running_corrects = 0

		for images, labels in train_loader:
			images = images.float().to(params.DEVICE)
			labels = labels.to(params.DEVICE)
			
			mappedLabels = mapFunction(labels, col)
			onehot_labels = torch.eye(10)[mappedLabels].to(params.DEVICE)#it creates the one-hot-encoding list for the labels; needed for BCELoss
			
			optimizer.zero_grad() # Zero-ing the gradients
			
			# Forward pass to the network
			old_outputs = old_resNet(images)
			outputs = resNet(images)
			#classLoss, distLoss = calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits )
			loss = calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits )
			
			#loss = classLoss + distLoss
			
			# Get predictions
			
			cut_outputs = np.take_along_axis(outputs, col[None, :], axis = 1)
			
			_, preds = torch.max(cut_outputs.data, 1)
			
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
	t_l = 0
	resNet = torch.load('resNet_task' + str(task + 10) + '.pt')
	resNet.eval() # Set Network to evaluation mode
	running_corrects = 0
	
	col = []
	#in fase di test verifico su tutti le classi viste fino ad ora, quindi prendo da test splits gli indici dei gruppi da 0 a task
	for i,x in enumerate( test_splits[ :int(task/10) + 1]):
		 v = np.array(x)
		 col = np.concatenate( (col,v), axis = None)
	col = col.astype(int)	
	print(col)
	for images, labels in test_loader:
		images = images.float().to(params.DEVICE)
		labels = labels.to(params.DEVICE)
		
		mappedLabels = mapFunction(labels, col)
		
		onehot_labels = torch.eye(task + params.TASK_SIZE)[mappedLabels].to(params.DEVICE) #it creates the one-hot-encoding list for the labels; neede for BCELoss
		# Forward Pass
		outputs = resNet(images)
		# Get predictions
		
		cut_outputs = np.take_along_axis(outputs, col[None, :], axis = 1)
		#cut_outputs = outputs[..., 0 : task + params.TASK_SIZE]
		_, preds = torch.max(cut_outputs.data, 1)
		# Update Corrects
		print(mappedLabels)
		print(preds)
		print(running_corrects)
		running_corrects += torch.sum(preds == mappedLabels.data).data.item()
		t_l += len(images)
	# Calculate Accuracy
	accuracy = running_corrects / float(t_l)
	
	#Calculate Loss
	loss = F.binary_cross_entropy_with_logits(cut_outputs,onehot_labels)
	print('Validation Loss: {} Validation Accuracy : {}'.format(loss.item(),accuracy) )
	return(accuracy, loss.item())	  


def calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits):
	m = nn.Sigmoid()
	
	outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), onehot_labels.to(params.DEVICE)
	#cut_outputs = outputs[..., task : task + params.TASK_SIZE]
	col1 = np.array(train_splits[int(task/10)]).astype(int)
	
	col2 = []
	for i,x in enumerate( train_splits[ :int(task/10) ]):
		v = np.array(x)
		col2 = np.concatenate( (col2,v), axis = None)
	col2 = col2.astype(int)

	col3 = []
	for i,x in enumerate( train_splits[ :int(task/10) +1]):
		v = np.array(x)
		col3 = np.concatenate( (col3,v), axis = None)
	col3 = col3.astype(int)

	if( task == 0):
		cut_outputs = np.take_along_axis(outputs, col1[None, :], axis = 1)	
		loss = F.binary_cross_entropy_with_logits(cut_outputs,onehot_labels)
		
	if( task > 0 ):
		cut_outputs = np.take_along_axis(outputs, col3[None, :], axis = 1)
		out = np.take_along_axis(old_outputs, col2[None, :], axis = 1)
		target = torch.cat((out, onehot_labels), 1)
		#distLoss = F.binary_cross_entropy_with_logits( input=outputs[..., :task], target=m(old_outputs[..., :task]) )
		loss = F.binary_cross_entropy_with_logits( input=cut_outputs, target=m(target) )

	#distLoss = distLoss * (step-1)/step
	
	distLoss = torch.zeros(1, requires_grad=False).to(params.DEVICE)
	#print(f'class loss = {classLoss}' f' dist loss = {distLoss.item()}')
	return loss
