import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from PIL import Image
from DatasetCIFAR import params
from DatasetCIFAR import utils
from DatasetCIFAR import ResNet
from DatasetCIFAR import data_set
from DatasetCIFAR.data_set import Dataset 
from torchvision import transforms
from torch.utils.data import Subset, DataLoader

class ICaRLStruct (nn.Module):
  def __init__(self, n_classes = 100, dataset = None):
    super(ICaRLStruct, self).__init__()
    self.features_extractor = ResNet.resnet32(num_classes = 10)
    self.classifier = nn.Linear(self.features_extractor.fc.out_features, n_classes)

    self.k = 2000
    self.exemplars = [None]*n_classes #lista di vettori; ha 100 elementi, ognuno di dimensione m che contiene gli m examplars
    self.m = 0
    self.dataset = dataset
    self.exemplar_means = None
    #Costruisco exemplar come un vettore di liste: 
    #ogni elemento corrisponde alla lista di exemplar presente per quella specifica classe (l'indice di exemplar indica la classe)
    #ogni lista avrà dimentsione M (variante di task in task dunque)
    #Così per ottenere la lista di exemplar in analisi ogni volta posso usare col come con LWF
    self.means = {}
    self.cuda()

  def forward(self, x):
    x = self.features_extractor(x)
    x = self.classifier(x)
    return x


  def generateExemplars(self, images, m, idxY):
    '''
    images --> indexes of image from a class (Y) belonging to dataSet
    m --> num of elements to be selected for the class Y 
    '''
    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),])
    features = []

    for idx in images:
      self.cuda()
      img = self.dataset._data[idx]
      img = Variable(transform(Image.fromarray(img))).cuda()
      feature = self.features_extractor(img.unsqueeze(0)).data.cpu().numpy()
      '''
      unsqueeze need to allow the network to read a single image. It create a "false" new dimension that simulates the following:
      (n samples, channels, height, width).
      Dunque anche il valore di feature sarà una matrice, ideata per contenere i valori di tutto il batch. 
      Per questo motivo, nella seguente istruzione prendo solo il primo elemento ([0])
      '''
      features.append(feature[0]) 
    features = np.array(features)
    mu = np.mean(features, axis=0)
    phiExemplaresY = []
    exemplaresY = []
    for k in range(0,m): #k parte da 0: ricorda di sommare 1 per media
      phiX = features #le features di tutte le immagini della classe Y
      phiP = np.sum(phiExemplaresY, axis = 0) #ad ogni step K, ho già collezionato K-1 examplars
      mu1 = 1/(k+1)* ( phiX + phiP)
      idxEx = np.argmin(np.sqrt(np.sum((mu - mu1) ** 2, axis=1))) #execute the euclidean norm among all the rows in phiX

      exemplaresY.append(images[idxEx])
      phiExemplaresY.append(features[idxEx])
    #Put into the exemplar array, at position related to the Y class, the elements obtained during this task
    self.exemplars[idxY] = np.array(exemplaresY)
    print('len exemplars[', idxY, '] = ', len(self.exemplars[idxY]))


  def reduceExemplars(self, m):
    for i in range(0, len(self.exemplars)):
      if(self.exemplars[i] is not None):
        self.exemplars[i] = np.array(self.exemplars[i])[:m]


  def updateRep(self, task, trainDataSet, splits, transformer):
    #torch.cuda.empty_cache()
    '''
    trainDataSet is the subset obtained by extracting all the data having as label those contained into train splits
    '''
    D = Dataset(transform=transformer)
    print(f'task = {task} ')
    #Define the parameters for traininig:
    optimizer = torch.optim.SGD(self.parameters(), lr=params.LR, momentum=params.MOMENTUM, weight_decay=params.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA) #allow to change the LR at predefined epochs
    current_step = 0
    with torch.no_grad():
      for y in splits:
        y = int(y)
        if(self.exemplars[y] is not None):
          length = len(self.exemplars[y])
          exLabels = [y]*length #dovrebbe crearmi un vettore di dimensione lenght tutto composto da y ovvero la classe
          exImages = self.exemplars[y]
          D.append(exImages, exLabels)

    loader = DataLoader( D, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle = True)
    #Now D contains both images and examplers for classes in analysis
    old_outputs = torch.zeros( len(D), 100).to(params.DEVICE)
    #torch.cuda.empty_cache()
    with torch.no_grad():
      for img, lbl, idx in loader:
        img = img.float().to(params.DEVICE)
        idx = idx.to(params.DEVICE)
        old_outputs[idx,:] = self.forward(img)

    col = np.array(splits[int(task/10)]).astype(int)
    for epoch in range(params.NUM_EPOCHS):
      for images, labels, idx in loader:
        images = images.float().to(params.DEVICE)
        labels = labels.to(params.DEVICE)

        onehot_labels = torch.eye(100)[labels].to(params.DEVICE)

        optimizer.zero_grad()

        outputs = self.forward(images)
        loss = utils.calculateLoss(outputs, old_outputs[idx,:], onehot_labels, task, splits )
        loss.backward()  # backward pass: computes gradients
        optimizer.step()
      print('Task: ' , task, ' epoch: ', epoch, ' loss: ', loss.item())


  def classify(self, x, col):
    '''
    x -> [BATCH SIZE] images to be classified
    col -> list classes see until now
    '''
    print('col = ', col)
    with torch.no_grad() :
      transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),])
      examplars = self.exemplars
      phi = self.features_extractor
      exemplar_means = []
      for P_y in col: #itero per tutte le classi in analisi
        P_y = int(P_y)
        print('Y = ', P_y)
        features = []
        #in P_y io ho m elementi, ovvero gli exemplars per quella specifica classe
        if(self.exemplars[P_y] is not None):
          print('Not none P_y = ', P_y)
          #exemplar contine gli indici delle immagini di riferiemnto
          for ex in self.exemplars[P_y]:
            image, label, idx = self.dataset.__getitem__(ex)
            img = self.dataset._data[ex]
            img = Variable(transform(Image.fromarray(img))).cuda()

            feature = phi(img.unsqueeze(0)) #unsqueeze add a dimension; i need it because feat ext expects a vector of imgs, not a single img
            feature = feature.squeeze() # squeeze needed beacause phi return a matrix of features, on row for each img. But i have only 1 img
            feature.data /= feature.data.norm()
            features.append(feature)
          print('3:', len(features))
          features = torch.stack(features)
          print('4:', features.size())
          mu_y = features.mean(0).squeeze()
          mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
          print('5:', mu_y.size())
          exemplar_means.append(mu_y)
    self.exemplar_means = exemplar_means

    means = torch.stack(exemplar_means) 
    print('6: ', means.size())
    means = torch.stack([means] * len(x)) #meglio usare len(x) che batch size per ultimo batch?
    print('7: ', means.size())
    means = means.transpose(1, 2) 
    print('8: ', means.size())
    with torch.no_grad():
      feature = self.features_extractor(x) # (batch_size, feature_size)
    for i in range(feature.size(0)): # Normalize
      feature.data[i] = feature.data[i] / feature.data[i].norm()
    feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
    print('9: ', feature.size())
    feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)
    print('10: ', feature.size())
    
    dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
    _, preds = dists.min(1)

    return preds
