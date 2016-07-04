import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Tools.utilities import *
from Tools.pycocodata import *
import numpy as np

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *


def get_model_inputs(img_id, caption=''):
    return np.zeros(HIDDEN)

def get_lstm_inputs():
    inputs = InputLayer((BATCH_SIZE, HIDDEN))
    return (inputs.input_var, inputs)
