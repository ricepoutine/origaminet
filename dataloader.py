import os
import pandas as pd
import torchvision
import numpy as np
from PIL import Image
import itertools
from torch.utils.data import Dataset
import skimage

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

def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int( y * max_h / x ),max_w)
    x = max_h

    img = np.array(Image.fromarray(img).resize((y,x)))
    return img

#get images as an np array, needed to overwrite __getitem__
def get_images(filenames, max_w, max_h, nch):
    try:
        image_data = np.array(Image.open(filenames))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)
        
        if nch==3 and image_data.shape[2]!=3:
            image_data = np.tile(image_data,3)

        image_data = np.pad(image_data,((0,0),(0,max_w-np.shape(image_data)[1]),(0,0)), mode='constant', constant_values=(1.0))

    except IOError as e:
        print('Could not read:', filenames, ':', e)
    
    return image_data
#####################################################################################################