import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Tools.utilities import *
from Tools.pycocodata.data import *
import numpy as np
import pickle

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *

def compressCatids():
    catids = getCatids()
    cats = set()
    for _, arr in catids.iteritems():
        for i in arr:
            if i not in cats: cats.add(i)
    cats = list(cats)
    cmap = {}
    dmap = {}
    for n, i in enumerate(cats):
        cmap[i] = n
        dmap[n] = i
    return dict([('cmap', cmap), ('dmap', dmap)])

mapping = compressCatids()
def get_model_inputs(img_id, caption=''):
    ans = np.zeros((80, 32, 32), dtype=theano.config.floatX)
    file_path = DATA_FOLDER + '/preprocessed/segmentations/segmaps_' + str(img_id) + '.pickle'
    if os.path.isfile(file_path):
        f = open(file_path, 'r')
        segmaps = pickle.load(f)
        f.close()
        for obj_id, bitmap in segmaps.iteritems():
            ans[mapping['cmap'][obj_id]] = bitmap
    return ans

STRUCT = [80,80,0]
def get_lstm_inputs():
    inputs = InputLayer((None,80,32,32))
    input_var = inputs.input_var
    for s in STRUCT:
        if s == 0: inputs = MaxPool2DLayer(inputs, 2)
        else: inputs = Conv2DLayer(inputs, s, (3,3))
    return input_var, DenseLayer(ReshapeLayer(inputs, (-1, 15680)), 512)
