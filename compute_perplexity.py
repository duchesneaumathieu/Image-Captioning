from get_inputs import *
from Tools.utilities import *
from Tools.pycocodata import *

import numpy as np
import pickle

import sys
if len(sys.argv)!=2:
    print 'python ../compute_perplexity.py <model name>'
    sys.exit()

name = sys.argv[1]

tokcap = getTesTokcap()
dictio = getDictio()
permaps = {}

model = Model(get_lstm_inputs(), name)
cost = theano.function(model.inputs, model.xent)

def hms(sec):
    s = sec%60
    m = (sec/60)%60
    h = (sec/3600)
    return '%sh %sm %ss'%('%02d'%h, '%02d'%m, '%02d'%s)

timer = Timer()
for n, (cap_id, cap_map) in enumerate(tokcap.iteritems()):
    caption = cap_map['caption']
    img_id = cap_map['image_id']
    if img_id not in permaps: permaps[img_id] = {}
    y = np.asarray([sentence2onehot(dictio, 'BEG '+cap_map['caption']+' END')]*32, dtype=theano.config.floatX)
    permaps[img_id][cap_id] =  cost(np.asarray([get_model_inputs(cap_map['image_id'])]*32, dtype=theano.config.floatX), y)
    t = timer.time()
    sys.stdout.write('\r%s%% complete! Expected time: %s'%("%010.6f"%(100*(n+1)/float(len(tokcap))),
                                                           hms(int(t/float(n+1)*(len(tokcap)-n-1)))))
    sys.stdout.flush()

print ''

f = open(name + '_all_perplexity.pickle', 'w')
pickle.dump(permaps, f, 2)
f.close()
