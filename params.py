import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
import random

DEVICE = 'cuda' # 'cuda' or 'cpu'
BATCH_SIZE = 128
NUM_WORKERS = 100
TASK_SIZE = 10

NUM_EPOCHS = 60

WEIGHT_DECAY = 0.00001
LR = 2
STEP_SIZE = [20,30,40,50]
GAMMA = 1/5
SEED = 42

def returnSplits():
	el = np.linspace(0,99,100)
	splits  = [None] * 10
	for i in range(0,10):
		random.seed(SEED)
		n = random.sample(set(el), k=10)
		splits[i] = n
		el = list( set(el) - set(n) )
	return splits 
