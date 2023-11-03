import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):

    def __init__(self, in_channels , num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1)
        
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self,x):

        x = nn.ReLU(self.conv1(x))

        x = self.pool(x)

        x = nn.ReLU(self.conv2(x))

        x = x.reshape(x.shape[0],-1)

        x = self.fc1(x)


        return x

