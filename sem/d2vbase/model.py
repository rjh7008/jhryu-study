import re
import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import numpy as np
import math
import logging
from collections import Counter



class mlpdist(nn.Module):

  def __init__(self,hidden_size, embedding_size,output_size):
    super(mlpdist,self).__init__()
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.output_size = output_size

    self.fc1 = nn.Linear(self.embedding_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.output_size)

  def aforward(self,input):
    out = self.fc1(input)
    out = self.fc2(out)
    return out

  def forward(self,input1,input2):
    out1 = self.aforward(input1)
    out2 = self.aforward(input2)
    result = torch.exp( -torch.norm( out1-out2, dim=1))

    return result


