import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from nltk.corpus import sentiwordnet as sn


def trans_senti(text,vocab):
  ret=[]

  for i in text:
    temp=[]
    for j in i:
      wr=[0.0,0.0,0.0]
      w = vocab.itos[j]
      syn = list(sn.senti_synsets(w))
      for s in syn:
        wr[0]+=s.pos_score()
        wr[1]+=s.neg_score()
        wr[2]+=s.obj_score()
      if len(syn) >1:
        wr[0]=wr[0]/len(syn)
        wr[1]=wr[1]/len(syn)
        wr[2]=wr[2]/len(syn)
      temp.append(wr)
    ret.append(temp)
  return ret



def train(train_iter, dev_iter, model, args, vocab):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label

            feature.data.t_(), target.data.sub_(1)  # batch first, index align

            #print(type(feature))
            senti_embed = trans_senti(feature,vocab)
            senti_embed = Variable(torch.FloatTensor(senti_embed)).cuda()

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            
            optimizer.zero_grad()
            logit = model(feature,senti_embed)
            
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args,vocab)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args,vocab):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align

        senti_embed = trans_senti(feature,vocab)
        senti_embed = Variable(torch.FloatTensor(senti_embed)).cuda()

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature,senti_embed)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
