from __init__ import *
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *

def lstm_xent(model_dist, true_dist):
    true_rsh = T.reshape(true_dist[:, 1:], (-1, WORDS_SIZE))
    model_rsh = T.reshape(model_dist[:, 1:-1], (-1, WORDS_SIZE))
    xent = T.nnet.categorical_crossentropy(model_rsh, true_rsh)
    return T.mean(xent)