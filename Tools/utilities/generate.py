from onehot import number2onehot
import numpy as np

def beamsearch(model, dictio, inputs, beam=10, memory=32, maxlen=20, sampling=True, printing=False):
    def allnotend(ds):
        b = False
        for d in ds:
            if d['word']!='END': b = True
        return b

    def separate(ended, unended, memory):
        mix = ended+unended
        sortedmix = sorted(mix, key=lambda k: k['prob'])
        ended2 = []
        unended2 =[]
        for un in sortedmix[-memory:]:
            if un['word'] == 'END': ended2 += [un]
            else: unended2 += [un]
        return ended2, unended2

    def getInputs(ds):
        words = []
        cells = []
        hidds = []
        sps = []
        for i in range(len(ds)):
            ds[i]['word']
            dictio[ds[i]['word']]
            words += [number2onehot(8848, dictio[ds[i]['word']])]
            cells += [ds[i]['cell']]
            hidds += [ds[i]['hidd']]
            sps += [ds[i]]
        return np.asarray(words), np.asarray(cells), np.asarray(hidds), sps

    def addbranch(probs, cs, hs, sps, beam):
        new = []
        for prob, c, h, sp in zip(probs, cs, hs, sps):
            indices = np.argsort(prob)[-beam:]
            for indice in indices:
                new += [dict([('word', dictio[indice]),
                                 ('sentence', sp['sentence']+' '+dictio[indice]), 
                                 ('cell', c), 
                                 ('hidd', h), 
                                 ('prob', sp['prob']*prob[indice])])]
        return new

    model.reset()
    _, c, h = model.step([inputs],0,0)
    n=0
    unended = [dict([('word', 'BEG'), ('sentence', ''), ('cell', c[0]), ('hidd', h[0]), ('prob', 1)])]
    ended = []
    while(len(unended) > 0 and n < maxlen):
        n += 1
        words, cells, hidds, sps = getInputs(unended)
        probs, cs, hs = model.step(words, cells, hidds)
        unended = addbranch(probs, cs, hs, sps, beam)
        ended, unended = separate(ended, unended, memory)
        #print len(ended), len(unended), len(ended)+len(unended)
        #for end in ended:
            #print end['sentence'][1:-4]
            
    probs = np.asarray([end['prob'] for end in ended])
    probs = probs/np.sum(probs)
    if printing:
        for end, prob in zip(ended, probs):
            print end['sentence'][1:-4], prob
    if sampling:
        probs = np.asarray([end['prob'] for end in ended])
        probs = probs/np.sum(probs)
        return ended[np.argmax(np.random.multinomial(1, probs))]['sentence'][1:-4]
    return ended[-1]['sentence'][1:-4]