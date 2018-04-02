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

from gensim import models

from model import mlpdist

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hidden_size", default=256, type=int, help='rnn hidden size'
  )
parser.add_argument('--embed', default = 300,type=int, help='embedding size'
  )
parser.add_argument('--output_size', default = 256,type=int, help='embedding size'
  )
parser.add_argument('--input',default='../stsbenchmark/sts-train.csv',help='path of training data'
  )
parser.add_argument('--dev',default='../stsbenchmark/sts-dev.csv',help='path of dev data'
  )
parser.add_argument('--epoch',default=100,type=int, help='number of epoch'
  )
parser.add_argument('--cuda',default=0,type=int,help='number of epoch'
  )
parser.add_argument('--batch',default=16, type=int, help='1:bidirectional, 0:non bidirectional'
  )

args = parser.parse_args()

d2v_model = models.Doc2Vec.load('semeval.d2v')

print (args)
net=None

cudanum = args.cuda
batch_size=args.batch
hidden_size=args.hidden_size
dt = []
dev = []


def load_data(filename,devfilename):
  from nltk.tokenize import word_tokenize
  for i,line in enumerate(open(filename,'r')):
    line=line.strip()

    item = line.split('\t')

    genre = item[0]
    filen = item[1]
    sent1 = item[5]
    sent2 = item[6]
    score = float(item[4])/5
    sent1 = word_tokenize(sent1)
    sent2 = word_tokenize(sent2)
    if genre != 'main-captions':
      continue
    dt.append( {'genre':genre,'filename':filen,'sent1':sent1,'sent2':sent2,'score':score } )

  for i,line in enumerate(open(devfilename,'r')):
    line = line.strip()
    item = line.split('\t')

    genre = item[0]
    filen = item[1]
    sent1 = item[5]
    sent2 = item[6]
    score = float(item[4])/5
    if score > 1:
      print('>5')
    sent1 = word_tokenize(sent1)
    sent2 = word_tokenize(sent2)
    if genre != 'main-captions':
      continue

    dev.append( {'genre':genre,'filename':filen,'sent1':sent1,'sent2':sent2,'score':score } )

def get_batch(i,dt):
  raw = dt[(i*batch_size):i*batch_size+batch_size]
  input1=[]
  input2=[]
  label=[]
  for it in raw:
    input1.append(d2v_model.infer_vector(it['sent1']).tolist())
    input2.append(d2v_model.infer_vector(it['sent2']).tolist())
    label.append(it['score'])
  return raw,input1,input2,label


first = True
def evalu():
  global first
  dev_loss = 0
  criterion = nn.MSELoss(size_average = False)

  for i_batch in range(0, math.ceil(len(dev)/batch_size)  ):
    loss = 0
    batch,i1,i2,score = get_batch(i_batch,dev)

    label = Variable(torch.FloatTensor(score)).cuda(cudanum)
    input1 = Variable(torch.FloatTensor(i1)).cuda(cudanum)
    input2 = Variable(torch.FloatTensor(i2)).cuda(cudanum)

    out = net(input1,input2)

    loss = criterion(out,label)
    dev_loss += loss.data[0]

    if i_batch < 1:
      print ('dev data first batch result')
      for i in range(len(input1)):
        if first:
          print ('sent1 :',' '.join(batch[i]['sent1']))
          print ('sent2 :',' '.join(batch[i]['sent2']))
        print ('predict : ',out.data[i],', ','label : ', label.data[i])
    if first:
      first=False
  print ('total dev loss : ',dev_loss)


def train(ep):
  random.shuffle(dt)
  random.shuffle(dev)

  criterion = nn.MSELoss(size_average=False)

  net_optim = optim.Adadelta(net.parameters())

  for i_epoch in range(ep):
    print(str(i_epoch) + ' epoch')
    total_loss =0
    for i_batch in range(0, math.ceil(len(dt)/batch_size)  ):
      net_optim.zero_grad()

      loss =0
      batch,i1,i2,score = get_batch(i_batch,dt)

      input1=Variable(torch.FloatTensor(i1)).cuda(cudanum)
      input2=Variable(torch.FloatTensor(i2)).cuda(cudanum)

      label = Variable(torch.FloatTensor(score)).cuda(cudanum)

      out = net(input1,input2)
      #print(out1)
      #print(out2)

      loss = criterion(out,label)
      total_loss += loss.data[0]

      loss.backward()
      net_optim.step()

    print ('train total loss : ',total_loss)
    evalu()

if __name__ == '__main__':
  load_data(args.input, args.dev)
  net= mlpdist(hidden_size = args.hidden_size,embedding_size = args.embed, output_size = args.output_size).cuda(cudanum)
  train(args.epoch)



