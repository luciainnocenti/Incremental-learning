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


def trainfunction(task, train_loader, train_splits):
	pars_epoch = [] #clean the pars_epoch after visualizations
	print(f'task = {task} ')
	resNet = torch.load('resNet_task' + str(task) + '.pt')
	old_resNet = torch.load('resNet_task' + str(task) + '.pt')

	#Define the parameters for traininig:
	optimizer = torch.optim.SGD(resNet.parameters(), lr=2.)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA) #allow to change the LR at predefined epochs
	current_step = 0

	##Train phase
	for epoch in range(params.NUM_EPOCHS):
		lenght = 0
		scheduler.step() #update the learning rate
		#print(scheduler.get_lr(), "   ", scheduler.get_last_lr()) #check if the lr is okay
		running_corrects = 0

		for images, labels in train_loader:
			images = images.float().to(params.DEVICE)
			labels = labels.to(params.DEVICE)
			
			onehot_labels = torch.eye(10)[labels%10].to(params.DEVICE)#it creates the one-hot-encoding list for the labels; needed for BCELoss
			
			optimizer.zero_grad() # Zero-ing the gradients
			
			# Forward pass to the network
			old_outputs = old_resNet(images)
			outputs = resNet(images)
			classLoss, distLoss = calculateLoss(outputs, old_outputs, onehot_labels, task)
			
			loss = classLoss + distLoss
			
			# Get predictions
			col = train_splits[int(task/10)]
			cut_outputs = np.take_along_axis(outputs, col[None, :], axis = 1)
			
			_, preds = torch.max(cut_outputs.data, 1)
			preds = preds + task
			# Update Corrects
			running_corrects += torch.sum(preds == labels.data).data.item()
			loss.backward()  # backward pass: computes gradients
			optimizer.step() # update weights based on accumulated gradients
			
			current_step += 1
			lenght += len(images)
		# Calculate Accuracy
		accuracy = running_corrects / float(lenght)
		print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ", accuracy)
	torch.save(resNet, 'resNet_task{0}.pt'.format(task + 10))


def calculateLoss(outputs, old_outputs, onehot_labels, task = 0):
	m = nn.Sigmoid()
	
	outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), onehot_labels.to(params.DEVICE)
	cut_outputs = outputs[..., task : task + params.TASK_SIZE]
	
	step = task/params.TASK_SIZE + 1
	classLoss = F.binary_cross_entropy_with_logits(cut_outputs,onehot_labels)
	classLoss /= step
	
	if( task > 0 ):
		distLoss = F.binary_cross_entropy_with_logits( input=outputs[..., :task], target=m(old_outputs[..., :task]) )
	else:
		distLoss = torch.zeros(1, requires_grad=False).to(params.DEVICE)
	
	distLoss = distLoss * (step-1)/step

	#print(f'class loss = {classLoss}' f' dist loss = {distLoss.item()}')
	return classLoss,distLoss

def evaluationTest(task, test_loader, test_splits):
	t_l = 0
	resNet = torch.load('resNet_task' + str(task + 10) + '.pt')
	resNet.eval() # Set Network to evaluation mode
	running_corrects = 0
	for images, labels in test_loader:
		images = images.float().to(params.DEVICE)
		labels = labels.to(params.DEVICE)
		onehot_labels = torch.eye(task + params.TASK_SIZE)[labels].to(params.DEVICE) #it creates the one-hot-encoding list for the labels; neede for BCELoss
		# Forward Pass
		outputs = resNet(images)
		# Get predictions
		col = train_splits[int(task/10)]
		cut_outputs = np.take_along_axis(outputs, col[None, :], axis = 1)
		#cut_outputs = outputs[..., 0 : task + params.TASK_SIZE]
		_, preds = torch.max(cut_outputs.data, 1)
		# Update Corrects
		running_corrects += torch.sum(preds == labels.data).data.item()
		t_l += len(images)
	# Calculate Accuracy
	accuracy = running_corrects / float(t_l)
	
	#Calculate Loss
	loss = F.binary_cross_entropy_with_logits(cut_outputs,onehot_labels)
	print('Validation Loss: {} Validation Accuracy : {}'.format(loss.item(),accuracy) )
	return(accuracy, loss.item())	  
 
def plotEpoch(pars):
	x_epochs = np.linspace(1,params.NUM_EPOCHS,params.NUM_EPOCHS)
	y1 = [e[0] for e in pars] #val acuracy
	y2 = [e[2] for e in pars] #train accuracy
	plt.plot(x_epochs, y1 , '-', color='red')
	plt.plot(x_epochs, y2, '-', color='blue')
	plt.xlabel("Epoch")
	plt.legend(['Validation Accuracy', 'Train accuracy'])
	plt.show()

	y1 = [e[1] for e in pars] #val loss
	y2 = [e[3] for e in pars] #train loss
	plt.plot(x_epochs, y1 , '-', color='red')
	plt.plot(x_epochs, y2, '-', color='blue')
	plt.xlabel("Epoch")
	plt.legend(['Validation Loss', 'Train Loss'])
	plt.show()

def plotTask(pars_tasks):
	x_tasks =  np.linspace(10, 100, 10)

	plt.plot(x_tasks, pars_tasks ,'b', label='Accuracy')
	plt.xlabel("Epoch")
	plt.title('Accuracy over classes')
	plt.legend(['Validation Accuracy'])
	plt.grid(True)
	plt.show()
