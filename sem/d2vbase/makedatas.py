fout = open('train_doc.txt','w')


from nltk.tokenize import word_tokenize
for i,line in enumerate(open('../stsbenchmark/sts-train.csv','r')):
  line=line.strip()
  item = line.split('\t')
  sent1 = item[5]
  sent2 = item[6]
  sent1 = word_tokenize(sent1)
  sent2 = word_tokenize(sent2)

  fout.write(' '.join(sent1) + '\n')
  fout.write(' '.join(sent2) + '\n')

