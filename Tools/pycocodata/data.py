from pycocotools.coco import COCO
import json
DATA_FOLDER='/u/duchema/Data/MsCoco'
COCO_TRAIN_OBJ_FILE='%s/annotations/instances_train2014.json'%(DATA_FOLDER)
COCO_TRAIN_CAP_FILE='%s/annotations/captions_train2014.json'%(DATA_FOLDER)
COCO_VALID_OBJ_FILE='%s/annotations/instances_val2014.json'%(DATA_FOLDER)
COCO_VALID_CAP_FILE='%s/annotations/captions_val2014.json'%(DATA_FOLDER)

def getTokcap(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/tokcap.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getLenids(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/lenids.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getDictio(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/dictio.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getCatids(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/catids.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getValimgids(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/valimgids.json'%(data_folder), 'r')
    d = json.load(f)
    f.close()
    return d

def getTesimgids(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/tesimgids.json'%(data_folder), 'r')
    d = json.load(f)
    f.close()
    return d

def getValTokcap(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/valtokcap.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getTesTokcap(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/testokcap.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getValLenids(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/vallenids.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d

def getFixSample(data_folder=DATA_FOLDER):
    f = open('%s/preprocessed/fixSample.json'%(data_folder), 'r')
    d = dict(json.load(f))
    f.close()
    return d
