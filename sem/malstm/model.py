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




class malstm(nn.Module):
  def __init__(self,embedding_size = 256, hidden_size = 256, vocab_size = 0):
    super(malstm,self).__init__()
    if vocab_size == 0:
      print ('vocab size is empty!')
      return -1
    self.vocab_size = vocab_size
    self.hidden_size= hidden_size
    self.embedding_size = embedding_size

    self.embed = nn.Embedding(self.vocab_size,self.embedding_size, padding_idx=0)

    self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = 1)


  def aforward(self, inputs, h, c):
    embed = self.embed(inputs).transpose(1,0)

    out,(oh, oc) = self.lstm(embed,(h,c))

    return out,oh,oc


  def forward(self,inputs1,inputs2, h, c):
    out1,_,__ = self.aforward(inputs1,h,c)
    out2,_,__ = self.aforward(inputs2,h,c)
    result = torch.exp( -torch.norm( out1[-1] - out2[-1], dim=1))

    return result



