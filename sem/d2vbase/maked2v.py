import gensim
from gensim import models
from gensim.models.doc2vec import LabeledSentence

filename = ''

doc_list=[]
labels_list=[]

for i,line in enumerate(open('train_doc.txt','r')):
  line=line.strip()
  doc_list.append(line)
  labels_list.append('TRAIN_'+str(i))

class LabeledLineSentence(object):
  def __init__(self,doc_list, labels_list):
    self.labels_list = labels_list
    self.doc_list = doc_list
  def __iter__(self):
    for i,doc in enumerate(self.doc_list):
      yield LabeledSentence(words=doc.split(' '),tags=[self.labels_list[i]])
  def __len__(self):
    return len(self.doc_list)


t = LabeledLineSentence(doc_list, labels_list)

model = models.Doc2Vec(vector_size=300 )
model.build_vocab(t)

model.train(t,epochs=15,total_examples=len(t))

model.save('semeval.d2v')

