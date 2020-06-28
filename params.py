import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
import random

DEVICE = 'cuda' # 'cuda' or 'cpu'
BATCH_SIZE = 128
TASK_SIZE = 10

NUM_EPOCHS = 2#70

WEIGHT_DECAY = 0.00001
LR = 2
STEP_SIZE = [49,63]
GAMMA = 1/5

SEED = 653

NUM_WORKERS = 4

K = 2000
MOMENTUM = 0.9



def returnSplits():
	el = np.linspace(0,99,100)
	splits  = [None] * 10
	for i in range(0,10):
		random.seed(SEED)
		n = random.sample(set(el), k=10)
		splits[i] = n
		el = list( set(el) - set(n) )
	return splits 

def returnSplits_v2():
	url = 'https://raw.githubusercontent.com/fcdl94/ICL/master/data/cifar_order.csv'
	data = pd.read_csv(url, header=None)

	splits = [None]*10
	classi = list(data.loc[9])
	for i in range(0,100,10):
	  splits[ int(i/10)] = classi[i:i+10]
	return splits
