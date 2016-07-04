from utilities import *
from pycocodata import *
from matplotlib.path import Path
import numpy as np
import json
import pickle

import sys
if len(sys.argv)!=3:
    print 'python ../sampling.py <ids file> <train or valid>'
    sys.exit()
    
ids_file = sys.argv[1]
tov = sys.argv[2]

f = open('./Splits/'+ids_file+'.json', 'r')
ids = json.load(f)
f.close()

if tov == 'train':
    obj = COCO(COCO_TRAIN_OBJ_FILE)
elif tov == 'valid':
    obj = COCO(COCO_VALID_OBJ_FILE)
else:
    sys.exit(0)

catids = getCatids()

def new_resolution(img, res):
    ans = np.zeros(res)
    for i, j in [(i, j) for i in range(res[0]) for j in range(res[1])]:
        bx = int(np.floor(i/float(res[0])*img.shape[0]))
        ex = int(np.ceil((i+1)/float(res[0])*img.shape[0]))
        by = int(np.floor(j/float(res[1])*img.shape[1]))
        ey = int(np.ceil((j+1)/float(res[1])*img.shape[1]))
        ans[i,j] = np.mean(img[bx:ex, by:ey])
    return ans

def get_segs_map(segs, res):
    ans = np.zeros(res)
    for seg in segs:
        ans += Path(seg).contains_points([(i, j) for j in range(res[0]) for i in range(res[1])]).reshape(res).astype(int)
    ans[np.where(ans > 1)] = 1
    return ans

def retreive_segs(objs, obj_id):
    segs = []
    for i in objs:
        if i['category_id'] == obj_id and not i['iscrowd']:
            seg = i['segmentation'][0]
            segs += [np.asarray(seg).reshape((len(seg)/2, 2))]
    return segs

def build_maps(img_id, new_res=None):
    objs = obj.imgToAnns[img_id]
    img = obj.loadImgs(img_id)[0]
    res = img['height'], img['width']
    if new_res is None: new_res = res
    
    maps = []
    for i in set(catids[img_id]):
        maps += [(i, new_resolution(get_segs_map(retreive_segs(objs, i), res), new_res))]
    return dict(maps)

def build_segmaps(img_ids, new_res=None):
    segmaps = []
    for img_id in img_ids:
        if img_id in obj.imgToAnns:
            segmaps += [(img_id, build_maps(img_id, new_res=new_res))]
    return dict(segmaps)

#next line is for debuging
#ids = ids[:10]
print 'Building segmaps...'
segmaps = build_segmaps(ids)

print 'Dumping...'
f = open(DATA_FOLDER + '/preprocessed/' + ids_file + '.pickle', 'w')
pickle.dump(segmaps, f, 2)
f.close()
