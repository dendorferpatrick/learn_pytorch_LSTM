
from __future__ import print_function

#from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import logging
import glob
from random import shuffle
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_txt_files(base_dir):
    return glob.iglob(f"{base_dir}/**/*.txt", recursive=True)

def load_data(phase="train", path_data='/usr/wiss/dendorfp/dvl/projects/trajnet/data'): 
        if phase=="train":
            path=os.path.join(path_data, "train")
        elif phase == "test":
            path= os.path.join(path_data, "test")
        else:
            print("no valid phase")
        
        scale=np.loadtxt(os.path.join(path_data, "scale_{}.txt".format(phase)))
        data = get_txt_files(path )
        data_raw=pd.DataFrame()
        i=0
        for csv_file in data:
            
            data_raw=data_raw.append(pd.read_csv(csv_file, header=None, delimiter=" "))
            logging.info("{} loaded".format(csv_file))
        i+=1
        data_raw.columns=['frame', 'ID', 'x', 'y']
        return data_raw, scale


class Dataset(Dataset):
    def __init__(self, seq_len,predict, phase="train", val_split= 0.2, shuffle_data=True):
        self.data_raw, self.scale=load_data(phase=phase)
        self.data_raw=self.data_raw
        self.max_frame=self.data_raw.groupby('ID')['frame'].count().max()
        self.seq_len     = seq_len
        self.predict   =predict
        if self.max_frame < self.seq_len + self.predict: 
            print("ERROR: not enough frames for look back and look forward. Choose smaller time windows. Max number of frames %s" % self.max_frame)
            
        self.ID= self.data_raw['ID'].unique()
        self.data=[]
        self.data_Y=[]
        self.frame=[]
        logger.debug("data loaded")

      
        for index, ind in enumerate(self.ID):#self.number_of_ind-)):
            df=self.data_raw[self.data_raw.ID==ind]    
            df_ind= df.values
            velocity=df_ind[1:,-2:]-df_ind[:-1,-2:]
            for i in range(np.maximum(0, len(df_ind) - self.seq_len- self.predict+1)):
            #self.frame.append(df_ind[i, 0])
                x=-1 + 2*np.divide((-self.scale[1, :2]+df_ind[i:i+self.seq_len, -2:]),(self.scale[0, :2]-self.scale[1, :2]))
                y=-1+2*np.divide((-self.scale[1, :2]+df_ind[i+self.seq_len:i+self.seq_len+self.predict, -2:]),(self.scale[0, :2]-self.scale[1, :2]))
                
                if i < np.maximum(0, len(df_ind) - self.seq_len - self.predict+1):
                    diff_x=-1 + 2*np.divide((-self.scale[1, 2:]+velocity[i:i+self.seq_len-1,  -2:]),(self.scale[0, 2:]-self.scale[1, 2:]))
                    diff_y=-1+2*np.divide((-self.scale[1, 2:]+velocity[i+self.seq_len-1:i+self.seq_len-1+self.predict, -2:]),(self.scale[0, 2:]-self.scale[1, 2:]))
                
                
                self.data.append([x,y, diff_x, diff_y])
        logger.debug("size data x {0} y {1} vx {2} vy {3}".format(np.shape(self.data[0][0]), np.shape(self.data[0][1]), np.shape(self.data[0][2]),np.shape(self.data[0][3])))
        if shuffle_data:
            shuffle(self.data)
        self.train=self.data[:int((1-val_split)*len(self.data))]
        self.val=self.data[int(val_split*len(self.data)):]
       
class Loader():
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        X = self.data[idx][0]
        Y = self.data[idx][1]
     #   frame= self.frame[idx]
        diff_X=self.data[idx][2]
        diff_Y= self.data[idx][3]

        sample = {'X': X, 'Y': Y, 'diff_X': diff_X, 'diff_Y': diff_Y}

        return sample
