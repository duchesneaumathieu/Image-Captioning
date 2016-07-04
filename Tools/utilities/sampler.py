import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pycocodata import *
from onehot import *
from theano import config
import numpy as np

class Sampler:
    def __init__(self, retriever):
        self.tokcap = getTokcap()
        self.lenids = getLenids()
        self.dictio = getDictio()
        self.fixSamp = getFixSample()
        self.retriever = retriever
        
    def randomLength(self):
        n = sum([len(i) for _,i in self.lenids.iteritems()])
        r = np.random.randint(n)
        for k,i in self.lenids.iteritems():
            if r < len(i): return k
            r -= len(i)
            
    def chooseN(self, N=32):
        ids = self.lenids[self.randomLength()]
        choices = dict()
        while(len(choices)<N):
            chosenId = np.random.choice(ids)
            if chosenId not in choices : choices[chosenId] = self.tokcap[chosenId]
        return choices
    
    def sample(self, N=32):
        x, y = [], []
        for k, m in self.chooseN(N=N).iteritems():
            x += [self.retriever(m['image_id'], m['caption'])]
            y += [sentence2onehot(self.dictio, 'BEG '+m['caption']+' END')]
        return np.asarray(x, dtype=config.floatX), np.asarray(y, dtype=config.floatX)
    
    def fixSample(self, n):
        x, y, vx, vy = [], [], [], []
        for k, m in self.fixSamp[n]['train'].iteritems():
            x += [self.retriever(m['image_id'], m['caption'])]
            y += [sentence2onehot(self.dictio, 'BEG '+m['caption']+' END')]
        for k, m in self.fixSamp[n]['valid'].iteritems():
            vx += [self.retriever(m['image_id'], m['caption'])]
            vy += [sentence2onehot(self.dictio, 'BEG '+m['caption']+' END')]
        return (np.asarray(x, dtype=config.floatX), np.asarray(y, dtype=config.floatX)), (np.asarray(vx, dtype=config.floatX), np.asarray(vy, dtype=config.floatX))
