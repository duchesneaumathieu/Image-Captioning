#UTILITIES
from global_variables import *
from onehot import *
from sampler import *
from model import *
from lstm_xent import *
from generate import *
import matplotlib.pyplot as plt
import os.path
import datetime

class Curves:
    def __init__(self, name, samp, cost):
        self.name = name
        self.samp = samp
        self.cost = cost
        self.extension = '_curves.numpy'
        self.train_curve, self.valid_curve = np.asarray([]), np.asarray([])
        self.push()
        self.saving = [self.train_curve, self.valid_curve]
        self.load()
        
    def current(self):
        return self.valid_curve[-1]
    
    def save(self):
        self.saving = [self.train_curve, self.valid_curve]
    
    def push(self):
        (x,y),(vx,vy) = self.samp()
        self.train_curve = np.append(self.train_curve, self.cost(x, y))
        self.valid_curve = np.append(self.valid_curve, self.cost(vx, vy))
        
    def dump(self):
        f = open(self.name+self.extension, 'w')
        np.save(f, self.saving)
        f.close()
        
    def load(self):
        if not os.path.isfile(self.name+self.extension):
            self.dump()
        else:
            f = open(self.name+self.extension, 'r')
            self.saving = np.load(f)
            [self.train_curve, self.valid_curve] = self.saving
            f.close()
            
    def show(self):
        plt.plot(self.train_curve)
        plt.plot(self.valid_curve)
        plt.legend(['train', 'valid'])
        plt.title(self.name)
        plt.ylabel('Cross Entropy')
        plt.show()
        
class Timer:
    def __init__(self):
        self.beg = datetime.datetime.now()
    
    def reset(self):
        self.beg = datetime.datetime.now()
    
    def time(self):
        return (datetime.datetime.now() - self.beg).seconds