from Tools.utilities import *

import sys
if len(sys.argv)!=2:
    print 'python ../show_curve.py <model name>'
    sys.exit()
    
name = sys.argv[1]

curves = Curves(name, lambda: ((0,0),(0,0)), lambda x, y: 0)
curves.show()