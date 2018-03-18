import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import numpy as np
import math



class siam(nn.Module):
  def __init__(self, vocab_size, hidden_size=256,embedding_size=256):
    super(siam,self).__init__()
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_size= embedding_size


    self.embed = nn.Embedding(self.vocab_size, self.embedding_size)
    self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers = 1)

  def aforward (self, inputs,hidden):
    output = self.embed(inputs).transpose(1,0)
    output,hidden = self.gru(output,hidden)
    return output[-1]

  def forward(self, input1, input2, hidden1):
    out1 = self.aforward(input1,hidden1)
    out2 = self.aforward(input2,hidden1)
    return out1,out2


