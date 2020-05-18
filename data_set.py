{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "data_set.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyNW20YSwxw5ZeSgPnf3LRwV",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/luciainnocenti/IncrementalLearning/blob/master/data_set.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vu44BZjMDN2t",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import numpy as np\n",
        "import torch\n",
        "from torchvision import transforms\n",
        "from torchvision import datasets\n",
        "\n",
        "class Dataset(torch.utils.data.Dataset):\n",
        "    '''\n",
        "    The class Dataset define methods and attributes to manage a CIFAR100 dataset\n",
        "    Attributes:\n",
        "        train = Bool, default value = True\n",
        "        Transform\n",
        "        Target_Transform\n",
        "\n",
        "        _dataset = contains the pythorch dataset CIFAR100 defined by hyperparameters passes by input\n",
        "        _targets = contains a lisf of 60000 elements, and each one referred to an image of the dataset. The value of each element is an integer in [0,99] that explicit the label for that image\n",
        "                    E.g. _targets[100] get the label of the 100th image in the dataset\n",
        "        _data = contains a list of 60000 imagest, each one represented by a [32]x[32]x[3] vector that define pixels \n",
        "        _labelNames = contains a list of 100 elements, each one represent a class; it maps integer indexes to human readable labels\n",
        "        \n",
        "    '''\n",
        "  def __init__(self, train = True, transform=None, target_transform=None):\n",
        "    self._train = train\n",
        "    self._dataset = datasets.cifar.CIFAR100( 'data', train=train, download=True, transform= transform, target_transform = target_transform )\n",
        "    self._targets = np.array(self._dataset.targets) #Essendo CIFAR100 una sottoclasse di CIFAR10, qui fa riferimento a quell'implementazione.\n",
        "    self._data = np.array(self._dataset.data)\n",
        "    self._labelNames = __getClassesNames__()\n",
        "\n",
        "  def __getIndexesGroups__(self, startIndex = 0):\n",
        "    #This method returns a list containing the indexes of all the images belonging to classes [starIndex, startIndex + 10]\n",
        "    indexes = []\n",
        "    self.searched_classes = np.linspace(startIndex, startIndex + 9, 10)\n",
        "    i = 0\n",
        "    for el in self._targets:\n",
        "      if (el in self.searched_classes):\n",
        "        indexes.append(i)\n",
        "      i+=1\n",
        "    return indexes\n",
        "\n",
        "  def __getClassesNames__(self):\n",
        "    #This method returns a list mapping the 100 classes into a human readable label. E.g. names[0] is the label that maps the class 0\n",
        "    names = []\n",
        "    classi = list(self._dataset.class_to_idx.keys())\n",
        "    for i in self.searched_classes:\n",
        "      names.append(classi[int(i)])\n",
        "    return names\n",
        "  \n",
        "  def __getitem__(self, idx):\n",
        "    #Given an index, this method return the image and the class corresponding to that index\n",
        "    image = self._data[idx]\n",
        "    label = self._targets[idx]\n",
        "    return image, label\n",
        "\n",
        "  def __len__(self):\n",
        "    return len(self._targets)"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}