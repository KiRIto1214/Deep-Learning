import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):

    def __init__(self, input_size,hidden_size,output_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size , hidden_size)

        self.i2o = nn.Linear(input_size + hidden_size , output_size)

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self ,input_tensor , hidden_tensor):


        combined = torch.cat((input_tensor + hidden_tensor), dim=1)


        out = self.i2o(combined)

        hidden  = self.i2h(combined)

        out = self.softmax(out)

        return out,hidden
    
    def init__hidden(self):

        return torch.zeros(1,self.hidden_size)