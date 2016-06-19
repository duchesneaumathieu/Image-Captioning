from Tools.utilities import *
from Tools.pycocodata import *
from get_inputs import *
import numpy as np
import json

import sys
if len(sys.argv)!=4:
    print 'python ../sampling.py <model name> <prefix> <#img>'
    sys.exit()
    
name = sys.argv[1]
prefix = sys.argv[2]
n_img = int(sys.argv[3])

model = Model(get_lstm_inputs(), name)
dictio = getDictio()

testids = np.random.permutation(getTesimgids())[:n_img]

print 'Sampling from the model...'
samples = []
for img_id in testids:
    inputs = get_model_inputs(img_id)
    samples += [(img_id, [dict([('caption', beamsearch(model, dictio, inputs))])])]

print 'Dumping...'
f = open(name + '_' + prefix + '_samples.json', 'w')
json.dump(samples, f)
f.close()