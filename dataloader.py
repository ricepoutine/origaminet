import os
import pandas as pd
import torchvision
import numpy as np
from PIL import Image
import itertools
from torch.utils.data import Dataset

class KGBDataLoader(Dataset):
    def __init__(self, datasoup, kgb_data_path, ralph=None, fmin=True, mln=None):
        self.fns = grab_files(datasoup, kgb_data_path)
        self.tlabels = grab_labels(self.fns)
        if ralph == None:
            alph  = get_alphabet(self.tlbls)
            self.ralph = dict (zip(alph.values(),alph.keys()))
            self.alph = alph
        else:
            self.ralph = ralph
        
        if mln != None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns   = np.asarray(self.fns)[filt].tolist()
    
    #length from files grabbed from grab_files()
    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, index):
        timgs = get_images(self.fns[index])
        
        return (timgs , self.tlbls[index])

#collect the soup from the soupkitchen and keep filenames
def grab_files(datasoup, kgb_data_path):
    filenames = open(datasoup, 'r').readlines()
    filenames = [ kgb_data_path + x.strip() for x in filenames ]
    return filenames

#label each file
def grab_labels(filenames):
    labels = []
    for id,image_file in enumerate(filenames):
        fn  = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r').read()
        lbl = ' '.join(lbl.split()) #remove linebreaks if present
            
        labels.append(lbl)

    return labels

#alphabetize them? Idfk man
def get_alphabet(labels):
    
    coll = ''.join(labels)     
    unq  = sorted(list(set(coll)))
    unq  = [''.join(i) for i in itertools.product(unq, repeat = 1)]
    alph = dict( zip( unq,range(len(unq)) ) )

    return alph

#get images as an np array, needed to overwrite __getitem__
def get_images(filenames):
    try:
        image_data = np.array(Image.open(filenames))

    except IOError as e:
        print('Could not read:', filenames, ':', e)
    
    return image_data
#####################################################################################################