import numpy as np
import theano
from theano import config

def number2onehot(n, i):
    return np.asarray([j==i*1. for j in range(n)], dtype=config.floatX)

def onehot2number(onehot):
    return np.argmax(onehot)

def defaultmapping(mapping, value):
    if value not in mapping : value = 'DFT'
    return mapping[value]

def sentence2onehot(mapping, sentence):
    words, l = sentence.split(), len(mapping)/2
    return [number2onehot(l, defaultmapping(mapping, word)) for word in words]

def onehot2sentence(mapping, onehots):
    l = len(onehots[0])
    words = [mapping[onehot2number(onehot)] for onehot in onehots]
    return str(' '.join(words))