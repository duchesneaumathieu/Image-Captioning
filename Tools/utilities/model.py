from __init__ import *
import os.path
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *

def build_model(inputs, words_inputs): #shape (BATCH_SIZE, OBJS_SIZE) and (BATCH_SIZE, None, WORDS_SIZE) expected
    words_2d = ReshapeLayer(words_inputs, (-1, WORDS_SIZE))
    words_emb_2d = DenseLayer(words_2d, HIDDEN, W=lasagne.init.Orthogonal(), b=None, nonlinearity=None)
    words_emb = ReshapeLayer(words_emb_2d, (BATCH_SIZE, -1, HIDDEN))
    lstm_in = ConcatLayer([ReshapeLayer(inputs, (BATCH_SIZE, 1, HIDDEN)), words_emb], axis=1)
    lstm_out = LSTMLayer(lstm_in, num_units=HIDDEN)
    lstm_out_2d = ReshapeLayer(lstm_out, (-1, HIDDEN))
    words_dist_2d = DenseLayer(lstm_out_2d, WORDS_SIZE, W=lasagne.init.Orthogonal(), b=None, nonlinearity=softmax)
    words_dist = ReshapeLayer(words_dist_2d, (BATCH_SIZE, -1, WORDS_SIZE))
    return words_dist

class Model:
    def __init__(self, x, name, mode='training'):
        theano_x, lasagne_x = x
        self.name = name
        self.extension = '_model.numpy'

        #inputs
        self.lasagne_x = lasagne_x
        self.lasagne_y = InputLayer((BATCH_SIZE, None, WORDS_SIZE))
        self.theano_x = theano_x
        self.theano_y = self.lasagne_y.input_var
        self.inputs = [self.theano_x, self.theano_y]

        #Building model
        BUILD_MODEL_PRINT()
        self.model_dist = build_model(self.lasagne_x, self.lasagne_y)
        self.xent = lstm_xent(get_output(self.model_dist), self.theano_y)
        
        #Params
        self.params = get_all_params(self.model_dist)
        self.saved_param_values = get_all_param_values(self.model_dist)
        self.load()
        
        #testing zone
        self.imagefunc = theano.function([self.theano_x], get_output(self.lasagne_x))
        self.wordfunc = lambda x: np.dot(x, self.params[-19].get_value())
        self.lstmfunc = LSTM(self.params[-18:-1]).function
        self.probfunc = lambda x: np.dot(x, self.params[-1].get_value())
        tmp = T.matrix()
        self.softmax = theano.function([tmp], T.nnet.softmax(tmp))
        self.start = True
        
    def step(self, x, c, h):
        if self.start:
            lstm_in = self.imagefunc(x)
            c = self.params[-3].get_value()*np.ones_like(lstm_in)
            h = self.params[-2].get_value()*np.ones_like(lstm_in)
            self.start = False
        else:
            lstm_in = self.wordfunc(x)
        
        lstm_out, c, h = self.lstmfunc(lstm_in, c, h)
        out = self.softmax(self.probfunc(lstm_out))
        return out, c, h
        
    def reset(self):
        self.start = True
    
    def save(self):
        self.saved_param_values = get_all_param_values(self.model_dist)
    
    def dump(self):
        f = open(self.name+self.extension, 'w')
        np.save(f, self.saved_param_values)
        f.close()
    
    def load(self):
        if not os.path.isfile(self.name+self.extension):
            NEW_MODEL_PRINT()
            self.dump()
        else:
            EXISTING_MODEL_PRINT()
            f = open(self.name+self.extension, 'r')
            self.saved_param_values = np.load(f)
            set_all_param_values(self.model_dist, self.saved_param_values)
            f.close()
            
class LSTM:
    def __init__(self, params):
        self.W_in_to_ingate = params[0] #
        self.W_hid_to_ingate = params[1] #
        self.b_ingate = params[2] #
        self.W_in_to_forgetgate = params[3] #
        self.W_hid_to_forgetgate = params[4] #
        self.b_forgetgate = params[5] #
        self.W_in_to_cell = params[6] #
        self.W_hid_to_cell = params[7] #
        self.b_cell = params[8] #
        self.W_in_to_outgate = params[9] #
        self.W_hid_to_outgate = params[10] #
        self.b_outgate = params[11] #
        self.W_cell_to_ingate = params[12] #
        self.W_cell_to_forgetgate = params[13] #
        self.W_cell_to_outgate = params[14] #
        self.cell_init = params[15].get_value() 
        self.hid_init = params[16].get_value()
        self.cell_current = params[15].get_value()
        self.hid_current = params[16].get_value()
        self.cell = T.matrix('cell')
        self.hid = T.matrix('hid')
        self.x = T.matrix('x')
        self.function = self.get_function()
        
    def get_function(self):
        it = T.nnet.sigmoid(T.dot(self.x, self.W_in_to_ingate) + 
                           T.dot(self.hid, self.W_hid_to_ingate) +
                           self.W_cell_to_ingate * self.cell + self.b_ingate)
        
        ft = T.nnet.sigmoid(T.dot(self.x, self.W_in_to_forgetgate) + 
                           T.dot(self.hid, self.W_hid_to_forgetgate) +
                           self.W_cell_to_forgetgate * self.cell + self.b_forgetgate)
        
        ct = ft * self.cell + it * T.tanh(T.dot(self.x, self.W_in_to_cell) + 
                                          T.dot(self.hid, self.W_hid_to_cell) + 
                                          self.b_cell)
        
        ot = T.nnet.sigmoid(T.dot(self.x, self.W_in_to_outgate) + 
                           T.dot(self.hid, self.W_hid_to_outgate) +
                           self.W_cell_to_outgate * ct + self.b_outgate)
        
        ht = ot * T.tanh(ct)
        
        return theano.function([self.x, self.cell, self.hid], [ht, ct, ht])
        
    def cmp(self, inc):
        ans, self.cell_current, self.hid_current = self.function(inc, self.cell_current, self.hid_current)
        return ans
    
    def reset(self):
        self.cell_current = self.cell_init
        self.hid_current = self.hid_init