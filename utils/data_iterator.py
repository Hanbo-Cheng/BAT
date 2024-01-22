import numpy
import random
import pickle as pkl
import gzip
import cv2


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def dataIterator(feature_file,label_file,align_file,dictionary,redictionary,batch_size,batch_Imagesize,maxlen,maxImagesize):
    
    fp_feature=open(feature_file,'rb')
    features=pkl.load(fp_feature)
    fp_feature.close()

    fp_label=open(label_file,'rb')
    labels=pkl.load(fp_label)
    fp_label.close()

    fp_align=open(align_file,'rb')
    aligns=pkl.load(fp_align)
    fp_align.close()

    ltargets = {}
    rtargets = {}
    relations = {}
    lpositions = {}
    rpositions = {}

    # map word to int with dictionary
    for uid, label_lines in labels.items():
        lchar_list = []
        rchar_list = []
        relation_list = []
        lpos_list = []
        rpos_list = []
        for line_idx, line in enumerate(label_lines):
            parts = line
            lchar = parts[0]
            lpos = parts[1]
            rchar = parts[2]
            rpos = parts[3]
            relation = parts[4]
            if dictionary.__contains__(lchar):
                lchar_list.append(dictionary[lchar])
            else:
                print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)

            if dictionary.__contains__(rchar):
                rchar_list.append(dictionary[rchar])
            else:
                print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
            
            lpos_list.append(int(lpos))
            rpos_list.append(int(rpos))  

            if line_idx != len(label_lines)-1:
                if redictionary.__contains__(relation):
                    relation_list.append(redictionary[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
            else:
                relation_list.append(0) # whatever which one to replace End relation
        ltargets[uid]=lchar_list
        rtargets[uid]=rchar_list
        relations[uid]=relation_list
        lpositions[uid] = lpos_list
        rpositions[uid] = rpos_list

    imageSize={}
    for uid,fea in features.items():
        if uid in ltargets:
            imageSize[uid]=fea.shape[0]*fea.shape[1]
        else:
            continue

    imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element

    feature_batch=[]
    llabel_batch=[]
    rlabel_batch=[]
    relabel_batch=[]
    align_batch=[]
    lpos_batch=[]
    rpos_batch=[]

    feature_total=[]
    llabel_total=[]
    rlabel_total=[]
    relabel_total=[]
    align_total=[]
    lpos_total=[]
    rpos_total=[]

    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if uid not in ltargets:
            continue
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        llab=ltargets[uid]
        rlab=rtargets[uid]
        relab=relations[uid]
        ali=aligns[uid]
        lp=lpositions[uid]
        rp=rpositions[uid]
        batch_image_size=biggest_image_size*(i+1)
        if len(llab)>maxlen:
            print ('this sentence length bigger than', maxlen, 'ignore')
        elif size>maxImagesize:
            print ('this image size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                llabel_total.append(llabel_batch)
                rlabel_total.append(rlabel_batch)
                relabel_total.append(relabel_batch)
                align_total.append(align_batch)
                lpos_total.append(lpos_batch)
                rpos_total.append(rpos_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                llabel_batch=[]
                rlabel_batch=[]
                relabel_batch=[]
                align_batch=[]
                lpos_batch=[]
                rpos_batch=[]
                feature_batch.append(fea)
                llabel_batch.append(llab)
                rlabel_batch.append(rlab)
                relabel_batch.append(relab)
                align_batch.append(ali)
                lpos_batch.append(lp)
                rpos_batch.append(rp)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                llabel_batch.append(llab)
                rlabel_batch.append(rlab)
                relabel_batch.append(relab)
                align_batch.append(ali)
                lpos_batch.append(lp)
                rpos_batch.append(rp)
                i+=1

    # last batch
    feature_total.append(feature_batch)
    llabel_total.append(llabel_batch)
    rlabel_total.append(rlabel_batch)
    relabel_total.append(relabel_batch)
    align_total.append(align_batch)
    lpos_total.append(lpos_batch)
    rpos_total.append(rpos_batch)

    print ('total ',len(feature_total), 'batch data loaded')

    return list(zip(feature_total,llabel_total,rlabel_total,relabel_total,align_total,lpos_total,rpos_total)),uidList

class BatchBucket():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, 
                 feature_file, label_file, align_file, dictionary, redictionary, mode='train', direction = "L2R",
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._label_file = label_file
        self._align_file = align_file
        self._dictionary_file = dictionary
        self._redictionary_file = redictionary
        self._use_all = use_all
        self._dict_load()
        if(direction == "L2R"):
            self._data_load()
        else:
            self._bidirection_data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()
        self._mode = mode

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])
        self._RelationLexicon = {}
        fp = open(self._redictionary_file)
        stuff = fp.readlines()
        fp.close()
        for l in stuff:
            w = l.strip().split()
            self._RelationLexicon[w[0]] = int(w[1])
        

    def _bidirection_data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._rtargets = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            L2R_label, R2L_label = label_lines

            lchar_list_L2R = []
            rchar_list_L2R = []
            relation_list_L2R = []
            lpos_list_L2R = []
            rpos_list_L2R = []


            lchar_list_R2L = []
            rchar_list_R2L = []
            relation_list_R2L = []
            lpos_list_R2L = []
            rpos_list_R2L = []

            # get L2R label
            for line_idx, line in enumerate(L2R_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list_L2R.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_L2R.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_L2R.append(int(lpos))
                rpos_list_L2R.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_L2R.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    


        

            for line_idx, line in enumerate(R2L_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list_R2L.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_R2L.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_R2L.append(int(lpos))
                rpos_list_R2L.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_R2L.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid] = [lchar_list_L2R, lchar_list_R2L]                                                                                                                                                                                                      
            self._rtargets[uid] = [rchar_list_L2R, rchar_list_R2L]
            self._relations[uid] = [relation_list_L2R, relation_list_R2L]
            self._lpositions[uid] = [lpos_list_L2R, lpos_list_R2L]
            self._rpositions[uid] = [rpos_list_L2R, rpos_list_R2L]
            
        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]

    def _data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._rtargets = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            lchar_list = []
            rchar_list = []
            relation_list = []
            lpos_list = []
            rpos_list = []
            for line_idx, line in enumerate(label_lines):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list.append(int(lpos))
                rpos_list.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid]=lchar_list
            self._rtargets[uid]=rchar_list
            self._relations[uid]=relation_list
            self._lpositions[uid] = lpos_list
            self._rpositions[uid] = rpos_list

        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]


    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        #print('Max:', max_h, max_w, max_l)
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        init_l = max_l
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        #print(max_h, max_w, max_l)
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l: 
                    keys.append([h, w, l, h * w * l, 0])
                    #print(keys[-1])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        #print(l)
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                batch_size = 1
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                #print(item, h, w, l)
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            ltarget_batch = []
            rtarget_batch = []
            relation_batch = []
            lposition_batch = []
            rposition_batch = []
            align_batch = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                #feature = self._cutout(feature)
                ltarget = self._ltargets[uid]
                rtarget = self._rtargets[uid]
                relation = self._relations[uid] 
                lposition = self._lpositions[uid]
                rposition = self._rpositions[uid]
                align = self._aligns[uid]

                fea_batch.append(feature)
                ltarget_batch.append(ltarget)
                rtarget_batch.append(rtarget)
                relation_batch.append(relation)
                lposition_batch.append(lposition)
                rposition_batch.append(rposition)
                align_batch.append(align)

                uid_batches.append(uid)

            batches.append((fea_batch, ltarget_batch, rtarget_batch, relation_batch, align_batch, lposition_batch, rposition_batch))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches

    def get_bidirection_batches(self):
        '''
        only utilized when training two-branch models
        '''
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            ltarget_batch_L2R = []
            rtarget_batch_L2R = []
            relation_batch_L2R = []
            lposition_batch_L2R = []
            rposition_batch_L2R = []
            align_batch_L2R = []

            ltarget_batch_R2L = []
            rtarget_batch_R2L = []
            relation_batch_R2L = []
            lposition_batch_R2L = []
            rposition_batch_R2L = []
            align_batch_R2L = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                #feature = self._cutout(feature)
                ltarget_L2R, ltarget_R2L = self._ltargets[uid]
                rtarget_L2R, rtarget_R2L  = self._rtargets[uid]
                relation_L2R, relation_R2L = self._relations[uid] 
                lposition_L2R, lposition_R2L = self._lpositions[uid]
                rposition_L2R, rposition_R2L = self._rpositions[uid]
                align_L2R, align_R2L = self._aligns[uid]

                fea_batch.append(feature) # feature 只需要添加一次
                ltarget_batch_L2R.append(ltarget_L2R)
                rtarget_batch_L2R.append(rtarget_L2R)
                relation_batch_L2R.append(relation_L2R)
                lposition_batch_L2R.append(lposition_L2R)
                rposition_batch_L2R.append(rposition_L2R)
                align_batch_L2R.append(align_L2R)

                ltarget_batch_R2L.append(ltarget_R2L)
                rtarget_batch_R2L.append(rtarget_R2L)
                relation_batch_R2L.append(relation_R2L)
                lposition_batch_R2L.append(lposition_R2L)
                rposition_batch_R2L.append(rposition_R2L)
                align_batch_R2L.append(align_R2L)

                uid_batches.append(uid)

            batches.append((fea_batch, [ltarget_batch_L2R, ltarget_batch_R2L], [rtarget_batch_L2R, rtarget_batch_R2L], [relation_batch_L2R, relation_batch_R2L], [align_batch_L2R, align_batch_R2L], [lposition_batch_L2R, lposition_batch_R2L], [rposition_batch_L2R, rposition_batch_R2L]))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches
    def _cutout(self, img):
        if self._mode != 'train':
            return img
        rand_seed =  numpy.random.uniform(0,1)
        if rand_seed <= 0.5:
            cutout_img = get_connected_components(img, 100., 0.5, 'column')
        else:
            cutout_img = img
        return cutout_img
    
    
class BatchBucket_ch_class():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, 
                 feature_file, label_file, align_file, dictionary, dictionary_cat, redictionary, mode='train', direction = "L2R",
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._label_file = label_file
        self._align_file = align_file
        self._dictionary_file = dictionary
        self._dictionary_file_cata = dictionary_cat
        self._redictionary_file = redictionary
        self._use_all = use_all
        self._dict_load()
        if(direction == "L2R"):
            self._data_load()
        else:
            self._bidirection_data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()
        self._mode = mode

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])
        self._RelationLexicon = {}
        fp = open(self._redictionary_file)
        stuff = fp.readlines()
        fp.close()
        for l in stuff:
            w = l.strip().split()
            self._RelationLexicon[w[0]] = int(w[1])

        fp = open(self._dictionary_file_cata)
        stuff = fp.readlines()
        fp.close()
        self._lexicon_cata = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon_cata[w[0]] = int(w[1])
        

    def _bidirection_data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._ltargets_cata = {}
        self._rtargets = {}
        self._rtargets_cata = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            L2R_label, R2L_label = label_lines

            lchar_list_L2R = []
            lchar_list_L2R_cata = []
            rchar_list_L2R = []
            rchar_list_L2R_cata = []
            relation_list_L2R = []
            lpos_list_L2R = []
            rpos_list_L2R = []


            lchar_list_R2L = []
            lchar_list_R2L_cata = []
            rchar_list_R2L = []
            rchar_list_R2L_cata = []
            relation_list_R2L = []
            lpos_list_R2L = []
            rpos_list_R2L = []

            # get L2R label
            for line_idx, line in enumerate(L2R_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list_L2R.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                
                if self._lexicon_cata.__contains__(lchar):
                    lchar_list_L2R_cata.append(self._lexicon_cata[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_L2R.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)

                if self._lexicon_cata.__contains__(rchar):
                    rchar_list_L2R_cata.append(self._lexicon_cata[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_L2R.append(int(lpos))
                rpos_list_L2R.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_L2R.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    


        

            for line_idx, line in enumerate(R2L_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]

                if self._lexicon.__contains__(lchar):
                    lchar_list_R2L.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)

                if self._lexicon_cata.__contains__(lchar):
                    lchar_list_R2L_cata.append(self._lexicon_cata[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_R2L.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                
                if self._lexicon_cata.__contains__(rchar):
                    rchar_list_R2L_cata.append(self._lexicon_cata[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_R2L.append(int(lpos))
                rpos_list_R2L.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_R2L.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid] = [lchar_list_L2R, lchar_list_R2L] 
            self._ltargets_cata[uid] = [lchar_list_L2R_cata, lchar_list_R2L_cata]                                                                                                                                                                                                      
            self._rtargets[uid] = [rchar_list_L2R, rchar_list_R2L]
            self._rtargets_cata[uid] = [rchar_list_L2R_cata, rchar_list_R2L_cata]    
            self._relations[uid] = [relation_list_L2R, relation_list_R2L]
            self._lpositions[uid] = [lpos_list_L2R, lpos_list_R2L]
            self._rpositions[uid] = [rpos_list_L2R, rpos_list_R2L]
            
        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]

    def _data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._ltargets_cata = {}
        self._rtargets = {}
        self._rtargets_cata = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            lchar_list = []
            lchar_list_cata = []
            rchar_list = []
            rchar_list_cata = []
            relation_list = []
            lpos_list = []
            rpos_list = []
            for line_idx, line in enumerate(label_lines):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                
                if self._lexicon_cata.__contains__(lchar):
                    lchar_list_cata.append(self._lexicon_cata[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)

                if self._lexicon.__contains__(rchar):
                    rchar_list.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)

                if self._lexicon_cata.__contains__(rchar):
                    rchar_list_cata.append(self._lexicon_cata[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list.append(int(lpos))
                rpos_list.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid]=lchar_list
            self._ltargets_cata[uid]=lchar_list_cata
            self._rtargets[uid]=rchar_list
            self._rtargets_cata[uid]=rchar_list_cata
            self._relations[uid]=relation_list
            self._lpositions[uid] = lpos_list
            self._rpositions[uid] = rpos_list

        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]


    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        #print('Max:', max_h, max_w, max_l)
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        init_l = max_l
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        #print(max_h, max_w, max_l)
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l: 
                    keys.append([h, w, l, h * w * l, 0])
                    #print(keys[-1])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        #print(l)
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                batch_size = 1
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                #print(item, h, w, l)
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            ltarget_batch = []
            ltarget_cata_batch = []
            rtarget_batch = []
            rtarget_cata_batch = []
            relation_batch = []
            lposition_batch = []
            rposition_batch = []
            align_batch = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                #feature = self._cutout(feature)
                ltarget = self._ltargets[uid]
                ltarget_cata = self._ltargets_cata[uid]
                rtarget = self._rtargets[uid]
                rtarget_cata = self._rtargets_cata[uid]
                relation = self._relations[uid] 
                lposition = self._lpositions[uid]
                rposition = self._rpositions[uid]
                align = self._aligns[uid]

                fea_batch.append(feature)
                ltarget_batch.append(ltarget)
                ltarget_cata_batch.append(ltarget_cata)
                rtarget_batch.append(rtarget)
                rtarget_cata_batch.append(rtarget_cata)
                relation_batch.append(relation)
                lposition_batch.append(lposition)
                rposition_batch.append(rposition)
                align_batch.append(align)

                uid_batches.append(uid)

            batches.append((fea_batch, ltarget_batch, ltarget_cata_batch, rtarget_batch, rtarget_cata_batch, relation_batch, align_batch, lposition_batch, rposition_batch))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches

    def get_bidirection_batches(self):
        '''
        only utilized when training two-branch models
        '''
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            ltarget_batch_L2R = []
            ltarget_batch_cata_L2R = []
            rtarget_batch_L2R = []
            rtarget_batch_cata_L2R = []
            relation_batch_L2R = []
            lposition_batch_L2R = []
            rposition_batch_L2R = []
            align_batch_L2R = []

            ltarget_batch_R2L = []
            ltarget_batch_cata_R2L = []
            rtarget_batch_R2L = []
            rtarget_batch_cata_R2L = []
            relation_batch_R2L = []
            lposition_batch_R2L = []
            rposition_batch_R2L = []
            align_batch_R2L = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                #feature = self._cutout(feature)
                ltarget_L2R, ltarget_R2L = self._ltargets[uid]
                ltarget_cata_L2R, ltarget_cata_R2L = self._ltargets_cata[uid]
                rtarget_L2R, rtarget_R2L  = self._rtargets[uid]
                rtarget_cata_L2R, rtarget_cata_R2L = self._rtargets_cata[uid]
                relation_L2R, relation_R2L = self._relations[uid] 
                lposition_L2R, lposition_R2L = self._lpositions[uid]
                rposition_L2R, rposition_R2L = self._rpositions[uid]
                align_L2R, align_R2L = self._aligns[uid]

                fea_batch.append(feature) # feature 只需要添加一次
                ltarget_batch_L2R.append(ltarget_L2R)
                ltarget_batch_cata_L2R.append(ltarget_cata_L2R)
                rtarget_batch_L2R.append(rtarget_L2R)
                rtarget_batch_cata_L2R.append(rtarget_cata_L2R)
                relation_batch_L2R.append(relation_L2R)
                lposition_batch_L2R.append(lposition_L2R)
                rposition_batch_L2R.append(rposition_L2R)
                align_batch_L2R.append(align_L2R)

                ltarget_batch_R2L.append(ltarget_R2L)
                ltarget_batch_cata_R2L.append(ltarget_cata_R2L)
                rtarget_batch_R2L.append(rtarget_R2L)
                rtarget_batch_cata_R2L.append(rtarget_cata_R2L)
                relation_batch_R2L.append(relation_R2L)
                lposition_batch_R2L.append(lposition_R2L)
                rposition_batch_R2L.append(rposition_R2L)
                align_batch_R2L.append(align_R2L)

                uid_batches.append(uid)

            batches.append((fea_batch, [ltarget_batch_L2R, ltarget_batch_R2L], [ltarget_batch_cata_L2R, ltarget_batch_cata_R2L],
                             [rtarget_batch_L2R, rtarget_batch_R2L], [rtarget_batch_cata_L2R, rtarget_batch_cata_R2L], 
                             [relation_batch_L2R, relation_batch_R2L], [align_batch_L2R, align_batch_R2L],
                               [lposition_batch_L2R, lposition_batch_R2L], 
                               [rposition_batch_L2R, rposition_batch_R2L]))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches
    def _cutout(self, img):
        if self._mode != 'train':
            return img
        rand_seed =  numpy.random.uniform(0,1)
        if rand_seed <= 0.5:
            cutout_img = get_connected_components(img, 100., 0.5, 'column')
        else:
            cutout_img = img
        return cutout_img


class BatchBucket_ch_emb():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, 
                 feature_file, label_file, align_file, dictionary, dictionary_cat_emb, redictionary, mode='train', direction = "L2R",
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._label_file = label_file
        self._align_file = align_file
        self._dictionary_file = dictionary
        self._dictionary_file_cata = dictionary_cat_emb
        self._redictionary_file = redictionary
        self._use_all = use_all
        self._dict_load()
        if(direction == "L2R"):
            self._data_load()
        else:
            self._bidirection_data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()
        self._mode = mode

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])
        self._RelationLexicon = {}
        fp = open(self._redictionary_file)
        stuff = fp.readlines()
        fp.close()
        for l in stuff:
            w = l.strip().split()
            self._RelationLexicon[w[0]] = int(w[1])

        # fp = open(self._dictionary_file_cata)
        # stuff = fp.readlines()
        # fp.close()

        fp=open(self._dictionary_file_cata, 'rb')
        self._lexicon_cata=pkl.load(fp)
        fp.close()
        self._lexicon_cata['</s>'] = self._lexicon_cata['<\s>'] 

        # self._lexicon_cata = {}
        # for l in stuff:
        #     w = l.strip().split()
        #     self._lexicon_cata[w[0]] = int(w[1])
        

    def _bidirection_data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._ltargets_cata = {}
        self._rtargets = {}
        self._rtargets_cata = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            L2R_label, R2L_label = label_lines

            lchar_list_L2R = []
            lchar_list_L2R_cata = []
            rchar_list_L2R = []
            rchar_list_L2R_cata = []
            relation_list_L2R = []
            lpos_list_L2R = []
            rpos_list_L2R = []


            lchar_list_R2L = []
            lchar_list_R2L_cata = []
            rchar_list_R2L = []
            rchar_list_R2L_cata = []
            relation_list_R2L = []
            lpos_list_R2L = []
            rpos_list_R2L = []

            # get L2R label
            for line_idx, line in enumerate(L2R_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list_L2R.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                
                if self._lexicon_cata.__contains__(lchar):
                    lchar_list_L2R_cata.append(self._lexicon_cata[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_L2R.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)

                if self._lexicon_cata.__contains__(rchar):
                    rchar_list_L2R_cata.append(self._lexicon_cata[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_L2R.append(int(lpos))
                rpos_list_L2R.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_L2R.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    


        

            for line_idx, line in enumerate(R2L_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]

                if self._lexicon.__contains__(lchar):
                    lchar_list_R2L.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)

                if self._lexicon_cata.__contains__(lchar):
                    lchar_list_R2L_cata.append(self._lexicon_cata[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_R2L.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                
                if self._lexicon_cata.__contains__(rchar):
                    rchar_list_R2L_cata.append(self._lexicon_cata[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_R2L.append(int(lpos))
                rpos_list_R2L.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_R2L.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid] = [lchar_list_L2R, lchar_list_R2L] 
            self._ltargets_cata[uid] = [lchar_list_L2R_cata, lchar_list_R2L_cata]                                                                                                                                                                                                      
            self._rtargets[uid] = [rchar_list_L2R, rchar_list_R2L]
            self._rtargets_cata[uid] = [rchar_list_L2R_cata, rchar_list_R2L_cata]    
            self._relations[uid] = [relation_list_L2R, relation_list_R2L]
            self._lpositions[uid] = [lpos_list_L2R, lpos_list_R2L]
            self._rpositions[uid] = [rpos_list_L2R, rpos_list_R2L]
            
        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]

    def _data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._ltargets_cata = {}
        self._rtargets = {}
        self._rtargets_cata = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            lchar_list = []
            lchar_list_cata = []
            rchar_list = []
            rchar_list_cata = []
            relation_list = []
            lpos_list = []
            rpos_list = []
            for line_idx, line in enumerate(label_lines):
                parts = line
                lchar = parts[0]
                
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                
                if self._lexicon_cata.__contains__(lchar):
                    lchar_list_cata.append(self._lexicon_cata[lchar])
                else:
                    lchar_list_cata.append(np.zeros(100).astype(np.float32))
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)

                if self._lexicon.__contains__(rchar):
                    rchar_list.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)

                if self._lexicon_cata.__contains__(rchar):
                    rchar_list_cata.append(self._lexicon_cata[rchar])
                else:
                    rchar_list_cata.append(np.zeros(100).astype(np.float32))
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list.append(int(lpos))
                rpos_list.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid]=lchar_list
            self._ltargets_cata[uid]=lchar_list_cata
            self._rtargets[uid]=rchar_list
            self._rtargets_cata[uid]=rchar_list_cata
            self._relations[uid]=relation_list
            self._lpositions[uid] = lpos_list
            self._rpositions[uid] = rpos_list

        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]


    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        #print('Max:', max_h, max_w, max_l)
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        init_l = max_l
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        #print(max_h, max_w, max_l)
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l: 
                    keys.append([h, w, l, h * w * l, 0])
                    #print(keys[-1])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        #print(l)
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                batch_size = 1
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                #print(item, h, w, l)
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            ltarget_batch = []
            ltarget_cata_batch = []
            rtarget_batch = []
            rtarget_cata_batch = []
            relation_batch = []
            lposition_batch = []
            rposition_batch = []
            align_batch = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                #feature = self._cutout(feature)
                ltarget = self._ltargets[uid]
                ltarget_cata = self._ltargets_cata[uid]
                rtarget = self._rtargets[uid]
                rtarget_cata = self._rtargets_cata[uid]
                relation = self._relations[uid] 
                lposition = self._lpositions[uid]
                rposition = self._rpositions[uid]
                align = self._aligns[uid]

                fea_batch.append(feature)
                ltarget_batch.append(ltarget)
                ltarget_cata_batch.append(ltarget_cata)
                rtarget_batch.append(rtarget)
                rtarget_cata_batch.append(rtarget_cata)
                relation_batch.append(relation)
                lposition_batch.append(lposition)
                rposition_batch.append(rposition)
                align_batch.append(align)

                uid_batches.append(uid)

            batches.append((fea_batch, ltarget_batch, ltarget_cata_batch, rtarget_batch, rtarget_cata_batch, relation_batch, align_batch, lposition_batch, rposition_batch))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches

    def get_bidirection_batches(self):
        '''
        only utilized when training two-branch models
        '''
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            ltarget_batch_L2R = []
            ltarget_batch_cata_L2R = []
            rtarget_batch_L2R = []
            rtarget_batch_cata_L2R = []
            relation_batch_L2R = []
            lposition_batch_L2R = []
            rposition_batch_L2R = []
            align_batch_L2R = []

            ltarget_batch_R2L = []
            ltarget_batch_cata_R2L = []
            rtarget_batch_R2L = []
            rtarget_batch_cata_R2L = []
            relation_batch_R2L = []
            lposition_batch_R2L = []
            rposition_batch_R2L = []
            align_batch_R2L = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                #feature = self._cutout(feature)
                ltarget_L2R, ltarget_R2L = self._ltargets[uid]
                ltarget_cata_L2R, ltarget_cata_R2L = self._ltargets_cata[uid]
                rtarget_L2R, rtarget_R2L  = self._rtargets[uid]
                rtarget_cata_L2R, rtarget_cata_R2L = self._rtargets_cata[uid]
                relation_L2R, relation_R2L = self._relations[uid] 
                lposition_L2R, lposition_R2L = self._lpositions[uid]
                rposition_L2R, rposition_R2L = self._rpositions[uid]
                align_L2R, align_R2L = self._aligns[uid]

                fea_batch.append(feature) # feature 只需要添加一次
                ltarget_batch_L2R.append(ltarget_L2R)
                ltarget_batch_cata_L2R.append(ltarget_cata_L2R)
                rtarget_batch_L2R.append(rtarget_L2R)
                rtarget_batch_cata_L2R.append(rtarget_cata_L2R)
                relation_batch_L2R.append(relation_L2R)
                lposition_batch_L2R.append(lposition_L2R)
                rposition_batch_L2R.append(rposition_L2R)
                align_batch_L2R.append(align_L2R)

                ltarget_batch_R2L.append(ltarget_R2L)
                ltarget_batch_cata_R2L.append(ltarget_cata_R2L)
                rtarget_batch_R2L.append(rtarget_R2L)
                rtarget_batch_cata_R2L.append(rtarget_cata_R2L)
                relation_batch_R2L.append(relation_R2L)
                lposition_batch_R2L.append(lposition_R2L)
                rposition_batch_R2L.append(rposition_R2L)
                align_batch_R2L.append(align_R2L)

                uid_batches.append(uid)

            batches.append((fea_batch, [ltarget_batch_L2R, ltarget_batch_R2L], [ltarget_batch_cata_L2R, ltarget_batch_cata_R2L],
                             [rtarget_batch_L2R, rtarget_batch_R2L], [rtarget_batch_cata_L2R, rtarget_batch_cata_R2L], 
                             [relation_batch_L2R, relation_batch_R2L], [align_batch_L2R, align_batch_R2L],
                               [lposition_batch_L2R, lposition_batch_R2L], 
                               [rposition_batch_L2R, rposition_batch_R2L]))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches
    def _cutout(self, img):
        if self._mode != 'train':
            return img
        rand_seed =  numpy.random.uniform(0,1)
        if rand_seed <= 0.5:
            cutout_img = get_connected_components(img, 100., 0.5, 'column')
        else:
            cutout_img = img
        return cutout_img
    
    
class BatchBucket_online_bi():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, 
                 feature_file, label_file, align_file, online_file, dictionary, redictionary, mode='train', direction = "L2R",
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._label_file = label_file
        self._align_file = align_file
        self._online_file = online_file
        self._dictionary_file = dictionary
        self._redictionary_file = redictionary
        self._use_all = use_all
        self._dict_load()
        if(direction == "L2R"):
            self._data_load()
        else:
            self._bidirection_data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()
        self._mode = mode

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])
        self._RelationLexicon = {}
        fp = open(self._redictionary_file)
        stuff = fp.readlines()
        fp.close()
        for l in stuff:
            w = l.strip().split()
            self._RelationLexicon[w[0]] = int(w[1])
        

    def _bidirection_data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_online=open(self._online_file,'rb')
        self._onlines=pkl.load(fp_online)
        fp_online.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._rtargets = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            L2R_label, R2L_label = label_lines

            lchar_list_L2R = []
            rchar_list_L2R = []
            relation_list_L2R = []
            lpos_list_L2R = []
            rpos_list_L2R = []


            lchar_list_R2L = []
            rchar_list_R2L = []
            relation_list_R2L = []
            lpos_list_R2L = []
            rpos_list_R2L = []

            # get L2R label
            for line_idx, line in enumerate(L2R_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list_L2R.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_L2R.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_L2R.append(int(lpos))
                rpos_list_L2R.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_L2R.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    


        

            for line_idx, line in enumerate(R2L_label):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list_R2L.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list_R2L.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list_R2L.append(int(lpos))
                rpos_list_R2L.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list_R2L.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid] = [lchar_list_L2R, lchar_list_R2L]                                                                                                                                                                                                      
            self._rtargets[uid] = [rchar_list_L2R, rchar_list_R2L]
            self._relations[uid] = [relation_list_L2R, relation_list_R2L]
            self._lpositions[uid] = [lpos_list_L2R, lpos_list_R2L]
            self._rpositions[uid] = [rpos_list_L2R, rpos_list_R2L]
            
        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]

    def _data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        if('MfrDB0104' in self._features.keys()):
            del self._features['MfrDB0104']
        fp_feature.close()

        fp_label=open(self._label_file, 'rb')
        labels=pkl.load(fp_label)
        fp_label.close()

        fp_align=open(self._align_file,'rb')
        self._aligns=pkl.load(fp_align)
        fp_align.close()

        self._ltargets = {}
        self._rtargets = {}
        self._relations = {}
        self._lpositions = {}
        self._rpositions = {}

        # map word to int with dictionary
        for uid, label_lines in labels.items():
            lchar_list = []
            rchar_list = []
            relation_list = []
            lpos_list = []
            rpos_list = []
            for line_idx, line in enumerate(label_lines):
                parts = line
                lchar = parts[0]
                lpos = parts[1]
                rchar = parts[2]
                rpos = parts[3]
                relation = parts[4]
                if self._lexicon.__contains__(lchar):
                    lchar_list.append(self._lexicon[lchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                    
                if self._lexicon.__contains__(rchar):
                    rchar_list.append(self._lexicon[rchar])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                    
                
                lpos_list.append(int(lpos))
                rpos_list.append(int(rpos))  


                if  self._RelationLexicon.__contains__(relation):
                    relation_list.append( self._RelationLexicon[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    

            self._ltargets[uid]=lchar_list
            self._rtargets[uid]=rchar_list
            self._relations[uid]=relation_list
            self._lpositions[uid] = lpos_list
            self._rpositions[uid] = rpos_list

        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._ltargets[uid])) for uid, fea in
                             self._features.items()]


    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        #print('Max:', max_h, max_w, max_l)
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        init_l = max_l
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        #print(max_h, max_w, max_l)
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l: 
                    keys.append([h, w, l, h * w * l, 0])
                    #print(keys[-1])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        #print(l)
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                batch_size = 1
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                #print(item, h, w, l)
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            online_fea_batch = []
            ltarget_batch = []
            rtarget_batch = []
            relation_batch = []
            lposition_batch = []
            rposition_batch = []
            align_batch = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                online_feature = self._onlines[uid]
                #feature = self._cutout(feature)
                ltarget = self._ltargets[uid]
                rtarget = self._rtargets[uid]
                relation = self._relations[uid] 
                lposition = self._lpositions[uid]
                rposition = self._rpositions[uid]
                align = self._aligns[uid]

                fea_batch.append(feature)
                ltarget_batch.append(ltarget)
                rtarget_batch.append(rtarget)
                relation_batch.append(relation)
                lposition_batch.append(lposition)
                rposition_batch.append(rposition)
                align_batch.append(align)

                uid_batches.append(uid)

            batches.append(((fea_batch, online_fea_batch), ltarget_batch, rtarget_batch, relation_batch, align_batch, lposition_batch, rposition_batch))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches

    def get_bidirection_batches(self):
        '''
        only utilized when training two-branch models
        '''
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            online_fea_batch = []
            ltarget_batch_L2R = []
            rtarget_batch_L2R = []
            relation_batch_L2R = []
            lposition_batch_L2R = []
            rposition_batch_L2R = []
            align_batch_L2R = []

            ltarget_batch_R2L = []
            rtarget_batch_R2L = []
            relation_batch_R2L = []
            lposition_batch_R2L = []
            rposition_batch_R2L = []
            align_batch_R2L = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                online_feature = self._onlines[uid]
                #feature = self._cutout(feature)
                ltarget_L2R, ltarget_R2L = self._ltargets[uid]
                rtarget_L2R, rtarget_R2L  = self._rtargets[uid]
                relation_L2R, relation_R2L = self._relations[uid] 
                lposition_L2R, lposition_R2L = self._lpositions[uid]
                rposition_L2R, rposition_R2L = self._rpositions[uid]
                align_L2R, align_R2L = self._aligns[uid]

                fea_batch.append(feature) # feature 只需要添加一次
                online_fea_batch.append(online_feature)
                ltarget_batch_L2R.append(ltarget_L2R)
                rtarget_batch_L2R.append(rtarget_L2R)
                relation_batch_L2R.append(relation_L2R)
                lposition_batch_L2R.append(lposition_L2R)
                rposition_batch_L2R.append(rposition_L2R)
                align_batch_L2R.append(align_L2R)

                ltarget_batch_R2L.append(ltarget_R2L)
                rtarget_batch_R2L.append(rtarget_R2L)
                relation_batch_R2L.append(relation_R2L)
                lposition_batch_R2L.append(lposition_R2L)
                rposition_batch_R2L.append(rposition_R2L)
                align_batch_R2L.append(align_R2L)

                uid_batches.append(uid)

            batches.append(([fea_batch, online_fea_batch], [ltarget_batch_L2R, ltarget_batch_R2L], [rtarget_batch_L2R, rtarget_batch_R2L], [relation_batch_L2R, relation_batch_R2L], [align_batch_L2R, align_batch_R2L], [lposition_batch_L2R, lposition_batch_R2L], [rposition_batch_L2R, rposition_batch_R2L]))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches
    def _cutout(self, img):
        if self._mode != 'train':
            return img
        rand_seed =  numpy.random.uniform(0,1)
        if rand_seed <= 0.5:
            cutout_img = get_connected_components(img, 100., 0.5, 'column')
        else:
            cutout_img = img
        return cutout_img
def get_connected_components(imgInput, min_contour_area, cutout_sample_rate, cutout_patch_type):
    imgInput = imgInput.transpose(1,2,0)
    newRet, binaryThreshold = cv2.threshold(imgInput,127,255,cv2.THRESH_BINARY_INV)

    # dilation
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,10))

    rectdilation = cv2.dilate(binaryThreshold, rectkernel, iterations = 1)

    outputImage_origin = imgInput.copy()
    outputImage = cv2.cvtColor(outputImage_origin, cv2.COLOR_GRAY2BGR)
    npaContours, npaHierarchy = cv2.findContours(rectdilation.copy(),        
                                                cv2.RETR_EXTERNAL,                 
                                                cv2.CHAIN_APPROX_SIMPLE)
    for npaContour in npaContours:
        [intX, intY, intW, intH] = cv2.boundingRect(npaContour)                     
        if cv2.contourArea(npaContour) > min_contour_area:
            
            imgROI = binaryThreshold[intY:intY+intH, intX:intX+intW]   

            subContours, subHierarchy = cv2.findContours(imgROI.copy(),        
                                                cv2.RETR_EXTERNAL,                 
                                                cv2.CHAIN_APPROX_SIMPLE)
            for subContour in subContours:
                if cv2.contourArea(subContour) >= min_contour_area:
                    [pointX, pointY, width, height] = cv2.boundingRect(subContour)

                    rand_seed =  numpy.random.uniform(0,1)
                    # add patch
                    if rand_seed <= cutout_sample_rate:
                        # import pdb; pdb.set_trace()
                        mask = numpy.ones((height, width), numpy.float32)
                        if cutout_patch_type == 'rectangle':
                            x_length = int(width / 2)
                            y_length = int(height / 2.5)
                            y = numpy.random.randint(height)
                            x = numpy.random.randint(width)
                            y1 = numpy.clip(y - y_length // 2, 0, height)
                            y2 = numpy.clip(y + y_length // 2, 0, height)
                            x1 = numpy.clip(x - x_length // 2, 0, width)
                            x2 = numpy.clip(x + x_length // 2, 0, width)

                            mask[y1:y2, x1:x2] = 0
                            
                            mask = numpy.expand_dims(mask, axis=2)
                            mask = numpy.repeat(mask, 3, axis=2)
                            
                            local_img = outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:]
                            # patch_color = numpy.ones(mask[:, x1:x2].shape, numpy.float32) * 128.
                            local_img = local_img * mask
                            # local_img[:, x1:x2] += patch_color
                            outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:] = local_img
                        
                        elif cutout_patch_type == 'row':
                            x_length = int(width)
                            y_length = int(height / 4)
                            
                            y = numpy.random.randint(height)

                            y1 = numpy.clip(y - y_length // 2, 0, height)
                            y2 = numpy.clip(y + y_length // 2, 0, height)

                            mask[y1:y2, :] = 0
                            mask = numpy.expand_dims(mask, axis=2)
                            mask = numpy.repeat(mask, 3, axis=2)
                            
                            local_img = outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:]
                            # patch_color = numpy.ones(mask[y1:y2, ].shape, numpy.float32) * 128.
                            local_img = local_img * mask
                            # local_img[y1:y2, :] += patch_color
                            outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:] = local_img
                        
                        elif cutout_patch_type == 'column':
                            x_length = int(width / 3)
                            y_length = int(height)

                            y = numpy.random.randint(height)
                            x = numpy.random.randint(width)

                            x1 = numpy.clip(x - x_length // 2, 0, width)
                            x2 = numpy.clip(x + x_length // 2, 0, width)

                            mask[:, x1:x2] = 0.
                            mask = numpy.expand_dims(mask, axis=2)
                            mask = numpy.repeat(mask, 3, axis=2)
                            
                            local_img = outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:]
                            # patch_color = numpy.ones(mask[:, x1:x2].shape, numpy.float32) * 128.
                            local_img = local_img * mask
                            # local_img[:, x1:x2] += patch_color
                            outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:] = local_img
                        elif cutout_patch_type == 'cross':
                            # add 'row' patch
                            y_length_row = int(height / 4)
                            y_row = numpy.random.randint(height)

                            y1_row = numpy.clip(y_row - y_length_row // 2, 0, height)
                            y2_row = numpy.clip(y_row + y_length_row // 2, 0, height)

                            # add 'column' patch
                            x_length_column = int(width / 3)
                            x_column = numpy.random.randint(width)

                            x1_column = numpy.clip(x_column - x_length_column // 2, 0, width)
                            x2_column = numpy.clip(x_column + x_length_column // 2, 0, width)

                            mask[y1_row:y2_row, :] = 0
                            mask[:, x1_column:x2_column] = 0.
                            mask = numpy.expand_dims(mask, axis=2)
                            mask = numpy.repeat(mask, 3, axis=2)

                            local_img = outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:]
                            # patch_color = numpy.ones(mask[y1:y2, ].shape, numpy.float32) * 128.
                            local_img = local_img * mask
                            # local_img[y1:y2, :] += patch_color
                            outputImage[intY+pointY:intY+pointY+height,intX+pointX:intX+pointX+width,:] = local_img

    
    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)
    outputImage = numpy.expand_dims(outputImage,axis=0)
    return outputImage

def dataIterator_test(feature_file,dictionary,redictionary,batch_size,batch_Imagesize,maxImagesize):
    
    fp_feature=open(feature_file,'rb')
    features=pkl.load(fp_feature)
    fp_feature.close()

    imageSize={}
    for uid,fea in features.items():
        imageSize[uid]=fea.shape[0]*fea.shape[1]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element

    feature_batch=[]

    feature_total=[]

    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        batch_image_size=biggest_image_size*(i+1)
        if size>maxImagesize:
            print ('this image size bigger than', maxImagesize, 'ignore')
        elif uid == '34_em_225':
            print ('this image ignore', uid)
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                feature_batch.append(fea)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                i+=1

    # last batch
    feature_total.append(feature_batch)

    print ('total ',len(feature_total), 'batch data loaded')

    return feature_total, uidList
