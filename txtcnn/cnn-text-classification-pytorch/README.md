
simple implementation of Sentiment Lexical-Augmented Convolutional Neural Networks for Sentiment Analysis
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8005542&tag=1

code is based on https://github.com/Shawn1993/cnn-text-classification-pytorch

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy
* nltk

in python
```
import nltk
nltk.download('sentiwordnet')
```


## usage
```
python3 main.py
```


if you do not want to use sentiment lexicon

```
python3 main.py -no_senti
```
but caluate sentiment embed, just does not concat output vector
so running speed is same

**check original cnn usage in ORIGINAL_README.md**

