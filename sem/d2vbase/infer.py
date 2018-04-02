from gensim import models

model = models.Doc2Vec.load('semeval.d2v')

print(model.infer_vector([ 'A', 'girl', 'is', 'styling', 'her', 'hair', '.' ]) )
