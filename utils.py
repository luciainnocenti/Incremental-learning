from DatasetCIFAR.data_set import Dataset 
from DatasetCIFAR import ResNet
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torch.nn import functional as F

DEVICE = 'cuda' # 'cuda' or 'cpu'
BATCH_SIZE = 128
NUM_WORKERS = 100
TASK_SIZE = 10
############################################
#NUM_EPOCHS = 70
NUM_EPOCHS = 5
############################################


WEIGHT_DECAY = 0.00001
LR = 2
STEP_SIZE = [49,63]
GAMMA = 1/5

def trainfunction(task, train_loader, test_loader, pars_tasks):
	pars_epoch = [] #clean the pars_epoch after visualizations

	resNet = torch.load('resNet_task' + str(task) + '.pt')
	old_resNet = torch.load('resNet_task' + str(task) + '.pt')

	#Define the parameters for traininig:
	optimizer = torch.optim.SGD(resNet.parameters(), lr=2., weight_decay=WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, STEP_SIZE, gamma=GAMMA) #allow to change the LR at predefined epochs

	##Train phase
	for epoch in range(NUM_EPOCHS):

		scheduler.step() #update the learning rate
		print(scheduler.get_lr(), "   ", scheduler.get_last_lr()) #check if the lr is okay
		running_corrects = 0

		for images, labels in train_loader:

			images = images.float().to(DEVICE)
			labels = labels.to(DEVICE)
			
			onehot_labels = torch.eye(100)[labels].to(DEVICE)#it creates the one-hot-encoding list for the labels; needed for BCELoss
			
			optimizer.zero_grad() # Zero-ing the gradients
			
			# Forward pass to the network
			old_outputs = old_resNet(images)
			outputs = resNet(images)
			
			classLoss, distLoss = calculateLoss(outputs, old_outputs, onehot_labels, task)
			
			loss = classLoss + distLoss

			# Get predictions
			_, preds = torch.max(outputs.data, 1)
			
			# Update Corrects
			running_corrects += torch.sum(preds == labels.data).data.item()
			
			loss.backward()  # backward pass: computes gradients
			optimizer.step() # update weights based on accumulated gradients
			
			current_step += 1
		# Calculate Accuracy
		accuracy = running_corrects / float(len(train_dataset))
		print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ", accuracy)
		
		#Some variables useful for visualization
		
		param=eachepochevaluation() #run the network in the validation set, it returns validation accuracy and loss 
		
		pars_epoch.append( (param[0], param[1], accuracy, loss.item()) )
		#pars_epoch -->   val_acc,  val_loss, train_acc,train_loss
		pars_tasks[int(task/10)] += param[0] # 
		#pars_task[task/10] --> contains the sum of all the accuracies obtained in a specific task

	plotEpoch(pars_epoch) 
	pars_tasks[int(task/10)] /= NUM_EPOCHS #make the average sum(accuracy)/num_epochs	
	torch.save(resNet, 'resNet_task{0}.pt'.format(task+1))

def calculateLoss(outputs, old_outputs, onehot_labels, task = 0):
	classLoss = nn.BCEWithLogitsLoss(outputs, onehot_labels)
	distLoss = nn.BCEWithLogitsLoss(outputs[..., :task], old_outputs[..., :task]) if task else 0 #se task != 0, calcola la loss; altrimenti ritorna 0

	return classLoss,distLoss

def eachepochevaluation(task, test_loader):
	resNet = torch.load('resNet_task' + str(task) + '.pt')
	resNet.train(False) # Set Network to evaluation mode
	running_corrects = 0
	#confusion_matrix = torch.zeros(10, 10)
	for images, labels in test_loader:
		images = images.float().to(DEVICE)
		labels = labels.to(DEVICE)
		onehot_labels = torch.eye(100)[labels].to(DEVICE) #it creates the one-hot-encoding list for the labels; neede for BCELoss
		# Forward Pass
		outputs = resNet(images)
		# Get predictions
		_, preds = torch.max(outputs.data, 1)
		# Update Corrects
		running_corrects += torch.sum(preds == labels.data).data.item()
	
	# Calculate Accuracy
	accuracy = running_corrects / float(len(test_dataset))
	
	#Calculate Loss
	loss = nn.BCEWithLogitsLoss(outputs,onehot_labels)
	print("epoch =" + str(epoch))
	print('Validation Loss: {} Validation Accuracy : {}'.format(loss,accuracy))
	return (accuracy, loss.item())	  
 
