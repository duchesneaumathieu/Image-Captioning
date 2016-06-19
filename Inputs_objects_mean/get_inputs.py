import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Tools.utilities import *
from Tools.pycocodata.data import *
import numpy as np

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *

catids = getCatids()

def get_model_inputs(img_id, caption=''):
    ans = np.zeros(90)
    for n in catids[img_id]:
        ans += number2onehot(90, n-1)/len(catids[img_id])
    return ans

def get_lstm_inputs():
    objs_in = InputLayer((BATCH_SIZE, OBJS_SIZE))
    objs_emb = DenseLayer(objs_in, HIDDEN, W=lasagne.init.Orthogonal(), b=None, nonlinearity=None)
    return (objs_in.input_var, objs_emb)
