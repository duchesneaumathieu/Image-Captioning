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
n_img = sys.argv[3]

model = Model(get_lstm_inputs(), name)
dictio = getDictio()

if n_img == 'all': 
    testids = getTesimgids()
else:
    testids = np.random.permutation(getTesimgids())[:int(n_img)]

def hms(sec):
    s = sec%60
    m = (sec/60)%60
    h = (sec/3600)
    return '%sh %sm %ss'%('%02d'%h, '%02d'%m, '%02d'%s)

print 'Sampling from the model...'
samples = []
timer = Timer()
for n, img_id in enumerate(testids):
    inputs = get_model_inputs(img_id)
    samples += [(img_id, [dict([('caption', beamsearch(model, dictio, inputs))])])]
    t = timer.time()
    sys.stdout.write('\r%s%% complete! Expected time: %s'%("%010.6f"%(100*(n+1)/float(len(testids))),
                                                           hms(int(t/float(n+1)*(len(testids)-n-1)))))
    sys.stdout.flush()

print ''
print 'Dumping...'
f = open(name + '_' + prefix + '_samples.json', 'w')
json.dump(samples, f)
f.close()
