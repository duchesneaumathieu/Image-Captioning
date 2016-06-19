from get_inputs import *
from Tools.utilities import *
from Tools.pycocodata import *
import numpy as np

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *

import sys
if len(sys.argv)!=6:
    print 'python ../training.py <model name> <eta> <save check> <dump time> <n batch>'
    sys.exit()
    
name = sys.argv[1]
eta = float(sys.argv[2])
save_check = int(sys.argv[3])
dump_time = int(sys.argv[4])
n_batch = int(sys.argv[5])

model = Model(get_lstm_inputs(), name)
cost = theano.function(model.inputs, model.xent)
sampler = Sampler(get_model_inputs)
curves = Curves(name, lambda: sampler.fixSample(8), cost)
grad = theano.function(model.inputs, updates=rmsprop(model.xent, model.params, eta))

timer = Timer()
current_cost = curves.current()
best_cost = current_cost
print 'Training Started!'
for i in range(n_batch):
    for k in range(save_check):
        x, y = sampler.sample()
        grad(x, y)
    #Save check
    curves.push()
    current_cost = curves.current()
    if current_cost < best_cost:
        model.save()
        curves.save()
        best_cost = current_cost
    #Printing
    print_var = (i, best_cost, current_cost, curves.train_curve[-1])
    sys.stdout.write('\r%d loops completed, best cost:%f vs current cost:%f, current train cost:%f'%print_var)
    sys.stdout.flush()
    #Time check
    if timer.time() > dump_time:
        print ", Saving model..."
        model.dump()
        curves.dump()
        timer.reset()

model.dump()
curves.dump()
curves.show()
print ''