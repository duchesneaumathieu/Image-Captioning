from Tools.utilities import *
from Tools.pycocotools.coco import COCO
from Tools.pycocoevalcap.eval import COCOEvalCap
import numpy as np
import json

import sys
if len(sys.argv)!=2:
    print 'python ../testing.py <model name>'
    sys.exit()
    
name = sys.argv[1]

f = open(name + '_samples.json', 'r')
samples = dict(json.load(f))
f.close()


class Useless:
    def __init__(self, imgToAnns):
        self.imgToAnns = imgToAnns
        self.getImgIds = lambda : self.imgToAnns.keys()

useless = Useless(samples)
cap = COCO(COCO_VALID_CAP_FILE)

cocoEval = COCOEvalCap(cap, useless)
cocoEval.params['image_id'] = useless.getImgIds()
cocoEval.evaluate()