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
    self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = 1)

  def aforward (self, inputs,hidden,context):
    output = self.embed(inputs).transpose(1,0)
    output,(h,c) = self.lstm(output,(hidden,context))
    return output[-1]

  def forward(self, input1, input2, hidden1, context):
    out1 = self.aforward(input1,hidden1,context)
    out2 = self.aforward(input2,hidden1,context)
    return out1,out2

class ContrastiveLoss(nn.Module):
  def __init__(self,margin=2.0):
    super(ContrastiveLoss,self).__init__()
    self.margin = margin

  def forward(self,output1,output2,label):
    euclidean_distance = F.pairwise_distance(output1, output2)

    loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) 
    return loss_contrastive

