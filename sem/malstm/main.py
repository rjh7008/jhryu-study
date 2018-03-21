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
logger = logging.getLogger('main')
from model import malstm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hidden_size", default=256, help='rnn hidden size'
  )
parser.add_argument('--embed', default = 256, help='embedding size'
  )
parser.add_argument('--input',default='../stsbenchmark/sts-train.csv',help='path of training data'
  )
parser.add_argument('--dev',default='../stsbenchmark/sts-dev.csv',help='path of dev data'
  )
parser.add_argument('--epoch',default=10,help='number of epoch'
  )
parser.add_argument('--cuda',default=0,help='number of epoch'
  )
args = parser.parse_args()

net=None

cudanum = args.cuda

batch_size=16

hidden_size=args.hidden_size
dt = []
dev = []
vocab=0
vocab_size=0
vocab_name = 'train'

def save_pickle(obj, filename):
  if os.path.isfile(filename):
      logger.info("Overwriting %s." % filename)
  else:
      logger.info("Saving to %s." % filename)
  with open(filename, 'wb') as f:
      pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_dictionary(dt):
  vocab_counter = Counter()
  line_count =0
  from nltk.tokenize import word_tokenize

  for items in dt:
    sent1 = items['sent1']
    sent2 = items['sent2']

    for i in sent1:
      vocab_counter[i] +=1
    for i in sent2:
      vocab_counter[i]+=1
    line_count +=1 
  logger.info("Total: %d unique words with a total "
              "of %d words."
              % (len(vocab_counter), sum(vocab_counter.values())))

  if len(vocab_counter.most_common())>49998:
    vocab_cnt = vocab_counter.most_common(49998)
  else:
    vocab_cnt = vocab_counter
  vocab = {'__PAD__':0,'__UNK__': 1}

  for i,word in enumerate(vocab_cnt):
    vocab[word] = i+2 
  save_pickle(vocab_counter,'global_vocab.pkl')
  save_pickle(vocab,vocab_name+'_vocab.pkl')
  return vocab,len(vocab.keys())

def load_vocab():
  with open(vocab_name+'_vocab.pkl','rb') as f:
    v=pickle.load(f)
  return v



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
    dt.append( {'genre':genre,'filename':filen,'sent1':sent1,'sent2':sent2,'score':score } )

  vocab, vocab_size = make_dictionary(dt)
  
  for it in dt:
    sent1 = it['sent1']
    sent2 = it['sent2']
    it['sent1_idx'] = []
    for i in sent1:
      if i in vocab:
        it['sent1_idx'].append(vocab.get(i,0))
    it['sent2_idx'] = []
    for i in sent2:
      it['sent2_idx'].append(vocab.get(i,0) )


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

    dev.append( {'genre':genre,'filename':filen,'sent1':sent1,'sent2':sent2,'score':score } )

  for it in dev:
    sent1 = it['sent1']
    sent2 = it['sent2']
    it['sent1_idx'] = []
    for i in sent1:
      if i in vocab:
        it['sent1_idx'].append(vocab.get(i,0))
    it['sent2_idx'] = []
    for i in sent2:
      it['sent2_idx'].append(vocab.get(i,0) )
  
  return vocab,vocab_size
  
def get_batch(i,dt):
  raw = dt[(i*batch_size):i*batch_size+batch_size]
  input1=[]
  input2=[]
  label=[]
  for it in raw:
    input1.append(it['sent1_idx'])
    input2.append(it['sent2_idx'])
    label.append(it['score'])
  return raw,input1,input2,label

def make_padding(inp):
  vectorized_seqs = inp
  seq_lengths = torch.cuda.LongTensor(list(map(len, vectorized_seqs)))

  seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long().cuda(cudanum)

  for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = Variable(torch.LongTensor(seq))

  return seq_tensor


first = True
def evalu():
  global first
  dev_loss = 0
  criterion = nn.MSELoss(size_average = False)

  for i_batch in range(0, math.ceil(len(dev)/batch_size)  ):
    loss = 0
    batch,i1,i2,score = get_batch(i_batch,dev)

    label = Variable(torch.FloatTensor(score)).cuda(cudanum)
    input1 = make_padding(i1)
    input2 = make_padding(i2)


    hidden1 = Variable(torch.randn(1,len(input1),hidden_size)).cuda(cudanum)
    cont = Variable(torch.randn(1,len(input2),hidden_size)).cuda(cudanum)

    out = net(input1,input2,hidden1,cont)

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

      label = Variable(torch.FloatTensor(score)).cuda(cudanum)

      input1 = make_padding(i1)
      input2 = make_padding(i2)

      hidden1 = Variable(torch.randn(1,len(input1),hidden_size)).cuda(cudanum)
      cont = Variable(torch.randn(1,len(input2),hidden_size)).cuda(cudanum)
      
      out = net(input1,input2,hidden1,cont)
      #print(out1)
      #print(out2)

      loss = criterion(out,label)

      total_loss += loss.data[0]

      loss.backward()
      net_optim.step()

    print ('train total loss : ',total_loss)
    evalu()

if __name__ == '__main__':
  vocab,vocab_size = load_data(args.input, args.dev)
  print('voc',vocab_size)
  net= malstm(hidden_size = args.hidden_size,embedding_size = args.embed, vocab_size=vocab_size).cuda(cudanum)
  train(args.epoch)

