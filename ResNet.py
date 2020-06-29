import torch.nn as nn
import math
import torch
from DatasetCIFAR import params
"""
Credits to @hshustc
Taken from https://github.com/hshustc/CVPR19_Incremental_Learning/tree/master/cifar100-class-incremental
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, features = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if features:
            x = x / x.norm()
        else:
            x = self.fc(x)

        return x

def resnet20(pretrained=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

def resnet56(pretrained=False, **kwargs):
    n = 9
    model = ResNet(Bottleneck, [n, n, n], **kwargs)
    return model

class BiasLayer(nn.Module):
	def __init__(self):
		super(BiasLayer, self).__init__()
		self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
		self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
	def forward(self, x):
		return self.alpha * x + self.beta
	def printParam(self):
		print(self.alpha.item(), self.beta.item())


class BICModel():
	def __init__(self, splits):
		self.splits = splits
		self.bias_layer0 = BiasLayer().to(params.DEVICE)
		self.bias_layer1 = BiasLayer().to(params.DEVICE)
		self.bias_layer2 = BiasLayer().to(params.DEVICE)
		self.bias_layer3 = BiasLayer().to(params.DEVICE)
		self.bias_layer4 = BiasLayer().to(params.DEVICE)
		self.bias_layer5 = BiasLayer().to(params.DEVICE)
		self.bias_layer6 = BiasLayer().to(params.DEVICE)
		self.bias_layer7 = BiasLayer().to(params.DEVICE)
		self.bias_layer8 = BiasLayer().to(params.DEVICE)
		self.bias_layer9 = BiasLayer().to(params.DEVICE)
		self.bias_layers=[self.bias_layer0, self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5, self.bias_layer6, self.bias_layer7, self.bias_layer8, self.bias_layer9]
		
	def printBICparams(self):
		for el in self.bias_layers:
			el.printParam()
			
	def bias_forward(self, input):
		splits = self.splits
		in0 = input[:, :10]
		in1 = input[:, 10:20]
		in2 = input[:, 20:30]
		in3 = input[:, 30:40]
		in4 = input[:, 40:50]
		in5 = input[:, 50:60]
		in6 = input[:, 60:70]
		in7 = input[:, 70:80]
		in8 = input[:, 80:90]
		in9 = input[:, 90:100]
		#in0 = input[:, splits[0] ]
		#in1 = input[:, splits[1] ]
		#in2 = input[:, splits[3] ]
		out0 = self.bias_layer0(in0)
		out1 = self.bias_layer1(in1)
		out2 = self.bias_layer2(in2)
		out3 = self.bias_layer3(in3)
		out4 = self.bias_layer4(in4)
		out5 = self.bias_layer5(in5)
		out6 = self.bias_layer5(in6)
		out7 = self.bias_layer5(in7)
		out8 = self.bias_layer5(in8)
		out9 = self.bias_layer5(in9)
		return torch.cat([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9], dim = 1)

