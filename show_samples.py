from Tools.utilities import *

import sys
if len(sys.argv)!=2:
    print 'python ../show_samples.py <model name>'
    sys.exit()
    
name = sys.argv[1]

f = open(name + '_samples.json', 'r')
samples = dict(json.load(f))
f.close()

for key, item in samples.iteritems():
    print key, item[0]['caption']