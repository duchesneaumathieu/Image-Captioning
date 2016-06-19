from pycocodata.data import *
from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import numpy as np
import random
import json


def create_tokcap(data_folder=DATA_FOLDER):
    cap = COCO(COCO_TRAIN_CAP_FILE)
    
    listedCapMap = {}
    for i in cap.loadAnns(cap.getAnnIds()):
        listedCapMap[i['id']] = [dict([('caption',i['caption']), ('image_id', i['image_id'])])]
    tokenizedListedCapMap = PTBTokenizer().tokenize(listedCapMap)
    
    tokcap = [] #map caption ids to a map of its tokenized caption and image id
    for i, j in tokenizedListedCapMap.iteritems():
        tokcap += [(i, dict([('caption', j[0]), ('image_id', listedCapMap[i][0]['image_id'])]))]
    
    f = open(data_folder + '/preprocessed/tokcap.json', 'w')
    json.dump(tokcap, f)
    f.close()
    
    
def create_lenids(data_folder=DATA_FOLDER):
    tokcap = getTokcap()
    lenSortCapIds = {}
    for i,j in tokcap.iteritems():
        length = len(j['caption'].split())
        if length in lenSortCapIds : lenSortCapIds[length] += [i]
        else : lenSortCapIds[length] = [i]
    cleanLenSortCapIds = {}
    for i,j in lenSortCapIds.iteritems():
        if 32 <= len(lenSortCapIds[i]) : cleanLenSortCapIds[i] = j
               
    listedCleanLenSortCapIds = []
    for k, i in cleanLenSortCapIds.iteritems():
        listedCleanLenSortCapIds += [(k, i)]

    f = open(data_folder + '/preprocessed/lenids.json', 'w')
    json.dump(listedCleanLenSortCapIds, f)
    f.close()
    
    
def create_dictio(data_folder=DATA_FOLDER):
    tokcap = getTokcap()
    wordsCount = dict()
    for i,j in tokcap.iteritems():
        words = j['caption'].split()
        for w in words:
            if w not in wordsCount : wordsCount[w] = 1
            else : wordsCount[w] += 1

    words = set(['END', 'BEG', 'DFT'])
    for i,j in wordsCount.iteritems():
        if 5 <= j : words.add(i)

    dictio = list()
    for i,j in enumerate(words):
        dictio += [(i,j), (j,i)]
        
    f = open(data_folder + '/preprocessed/dictio.json', 'w')
    json.dump(dictio, f)
    f.close()
    

def creat_catids(data_folder=DATA_FOLDER):
    obj = COCO(COCO_TRAIN_OBJ_FILE)
    vobj = COCO(COCO_VALID_OBJ_FILE)
    catids = []
    for img_id in obj.getImgIds():
        if img_id in obj.imgToAnns:
            objs = []
            for obj_map in obj.imgToAnns[img_id]:
                objs += [obj_map['category_id']]
            catids += [(img_id, objs)]
        else : catids += [(img_id, [])]

    for img_id in vobj.getImgIds():
        if img_id in vobj.imgToAnns:
            objs = []
            for obj_map in vobj.imgToAnns[img_id]:
                objs += [obj_map['category_id']]
            catids += [(img_id, objs)]
        else : catids += [(img_id, [])]
            
    f = open(data_folder + '/preprocessed/catids.json', 'w')
    json.dump(catids, f)
    f.close()
    
    
def split_valid(data_folder=DATA_FOLDER):
    cap = COCO(COCO_VALID_CAP_FILE)
    imgIds = cap.getImgIds()
    random.seed(0)
    random.shuffle(imgIds)
    mid = len(imgIds)/2
    vimgids, timgids = imgIds[:mid], imgIds[mid:]

    f = open(data_folder + '/preprocessed/valimgids.json', 'w')
    json.dump(vimgids, f)
    f.close()
    
    f = open(data_folder + '/preprocessed/tesimgids.json', 'w')
    json.dump(timgids, f)
    f.close()
    

def create_valtokcap(data_folder=DATA_FOLDER):
    import gc
    gc.collect()
    vcap = COCO(COCO_VALID_CAP_FILE)
    valimgids, tesimgids = getValimgids(), getTesimgids()
    valcap = []
    for i in valimgids:
        valcap += vcap.imgToAnns[i]

    tescap = []
    for i in tesimgids:
        tescap += vcap.imgToAnns[i]
        
    vallistedCapMap = {}
    for i in valcap:
        vallistedCapMap[i['id']] = [dict([('caption',i['caption']), ('image_id', i['image_id'])])]
    valtokenizedListedCapMap = PTBTokenizer().tokenize(vallistedCapMap)

    teslistedCapMap = {}
    for i in tescap:
        teslistedCapMap[i['id']] = [dict([('caption',i['caption']), ('image_id', i['image_id'])])]
    testokenizedListedCapMap = PTBTokenizer().tokenize(teslistedCapMap)
    
    valtokcap = [] #map caption ids to a map of its tokenized caption and image id
    for i, j in valtokenizedListedCapMap.iteritems():
        valtokcap += [(i, dict([('caption', j[0]), ('image_id', vallistedCapMap[i][0]['image_id'])]))]

    testokcap = []
    for i, j in testokenizedListedCapMap.iteritems():
        testokcap += [(i, dict([('caption', j[0]), ('image_id', teslistedCapMap[i][0]['image_id'])]))]

    f = open(data_folder + '/preprocessed/valtokcap.json', 'w')
    json.dump(valtokcap, f)
    f.close()

    f = open(data_folder + '/preprocessed/testokcap.json', 'w')
    json.dump(testokcap, f)
    f.close()
    
    
def create_vallenids(data_folder=DATA_FOLDER):
    vcap = getValTokcap()
    tcap = getTesTokcap()
    vallenSortCapIds = {}
    for i,j in vcap.iteritems():
        length = len(j['caption'].split())
        if length in vallenSortCapIds : vallenSortCapIds[length] += [i]
        else : vallenSortCapIds[length] = [i]
    valcleanLenSortCapIds = {}
    for i,j in vallenSortCapIds.iteritems():
        if 32 <= len(vallenSortCapIds[i]) : valcleanLenSortCapIds[i] = j

    teslenSortCapIds = {}
    for i,j in tcap.iteritems():
        length = len(j['caption'].split())
        if length in teslenSortCapIds : teslenSortCapIds[length] += [i]
        else : teslenSortCapIds[length] = [i]
    tescleanLenSortCapIds = {}
    for i,j in teslenSortCapIds.iteritems():
        if 32 <= len(teslenSortCapIds[i]) : tescleanLenSortCapIds[i] = j
            
    vallistedCleanLenSortCapIds = []
    for k, i in valcleanLenSortCapIds.iteritems():
        vallistedCleanLenSortCapIds += [(k, i)]

    teslistedCleanLenSortCapIds = []
    for k, i in tescleanLenSortCapIds.iteritems():
        teslistedCleanLenSortCapIds += [(k, i)]
        
    f = open(data_folder + '/preprocessed/vallenids.json', 'w')
    json.dump(vallistedCleanLenSortCapIds, f)
    f.close()

    f = open(data_folder + '/preprocessed/teslenids.json', 'w')
    json.dump(teslistedCleanLenSortCapIds, f)
    f.close()
    
    
def create_fixSample(data_folder=DATA_FOLDER):
    np.random.seed(0)
    tokcap = getTokcap()
    valtokcap = getValTokcap()
    lenids = getLenids()
    vallenids = getValLenids()
    
    def chooseN(ids, tokcap, N=32):
        choices = dict()
        while(len(choices)<N):
            chosenId = np.random.choice(ids)
            if chosenId not in choices : choices[chosenId] = tokcap[chosenId]
        return choices
    
    fixSample = []
    for n in range(7,27):
        fixSample += [(n, dict([('train', chooseN(lenids[n], tokcap)), ('valid', chooseN(vallenids[n], valtokcap))]))]
        
    f = open(data_folder + '/preprocessed/fixSample.json', 'w')
    json.dump(fixSample, f)
    f.close()
    
    
if __name__ == '__main__':
    import os
    if not os.path.exists(DATA_FOLDER + '/preprocessed/'): os.makedirs(DATA_FOLDER + '/preprocessed/')
    print 'Creating tokcap...'
    create_tokcap()
    print 'Creating lenids...'
    create_lenids()
    print 'Creating dictio...'
    create_dictio()
    print 'Creating catids...'
    creat_catids()
    print 'Spliting valid...'
    split_valid()
    print 'Creating valtokcap...'
    create_valtokcap()
    print 'Creating vallenids...'
    create_vallenids()
    print 'Creating fixSample...'
    create_fixSample()