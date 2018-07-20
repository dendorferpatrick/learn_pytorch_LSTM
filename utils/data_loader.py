
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



class Dataset(Dataset):

    def __init__(self, csv_file,  seq_len,predict, phase='train'):
        self.data_raw      = pd.read_csv(csv_file, header=None, delimiter=" ")
        self.data_raw.columns=['frame', 'ID', 'x', 'y']
        #self.data_raw=self.data_raw[self.data_raw.lost==0]
        self.max_frame=self.data_raw.groupby('ID')['frame'].count().max()
        
        self.seq_len     = seq_len
        self.predict   =predict
        if self.max_frame < self.seq_len + self.predict: 
            print("ERROR: not enough frames for look back and look forward. Choose smaller time windows. Max number of frames %s" % self.max_frame)
            
        self.number_of_ind= self.data_raw['ID'].max()
        self.data_X=[]
        self.data_Y=[]
        self.frame=[]
        for index, ind in enumerate(np.arange(self.number_of_ind)):#self.number_of_ind-)):
            df=self.data_raw[self.data_raw.ID==ind]
            
            df_ind= df.values
            
            for i in range(np.maximum(0, len(df_ind) - self.seq_len- self.predict+1)):
                self.frame.append(df_ind[i, 0])
             
                self.data_X.append(df_ind[i:(i + self.seq_len), -2:])
                self.data_Y.append(df_ind[i + self.seq_len:(i + self.seq_len+self.predict), -2:])
              
        self.data_X=np.array(self.data_X).transpose(1,0,2)
        self.data_Y=np.array(self.data_Y).transpose(1,0,2)
    
        self.data_X=self.data_X*1./self.data_X.max()
      
        self.data_Y=self.data_Y*1./self.data_X.max()

        
        self.diff_X=self.data_X[:, 1:, :]-self.data_X[:, :-1, :]
        self.diff_Y=self.data_Y[:, 1:, :]-self.data_Y[:, :-1, :]

        self.max_diff= abs(self.diff_X).max()

        self.diff_X=self.data_X*1./self.max_diff
        self.diff_Y=self.data_Y*1./self.max_diff



    def __len__(self):
        return np.shape(self.data_X)[1]

    def __getitem__(self, idx):

        X = self.data_X[:, idx, :]
        Y = self.data_Y[:, idx, :]
        frame= self.frame[idx]
        diff_X=self.diff_X[:, idx, :]
        diff_Y= self.diff_Y[:, idx, :]

    

        
       
        sample = {'X': X, 'Y': Y, 'F': frame, 'diff_X': diff_X, 'diff_Y': diff_Y}

        return sample






"""
if __name__ == "__main__":


    path= 'data/centers_hyang_video0.csv'
    steps=300
    data= Dataset( path, steps, 3)
    loader = DataLoader(data, batch_size=1, num_workers=8, shuffle=True)
    import matplotlib.pyplot as plt
    import cv2
    path_video='videos/hyang/video0/video.mov'
    cap = cv2.VideoCapture(path_video)

    img_path='/Users/patrickdendorfer/phd/trajectory/annotations/hyang/video0/reference.jpg'
    for item , batch in enumerate(loader):
        cap = cv2.VideoCapture(path_video)
        cap.set(1,batch['F'][0])
        frame_nr=0
        plot=False 
        k=0
        scale=0.4
        print(batch['F'][0])
        while(True):
            ret, frame = cap.read()
            index=np.where(batch['F'].data.cpu().numpy()==frame_nr)[0]
            #print(batch['X'].size())
            
            plot=True
            
            if k==steps:
                    break
            if plot:
                coor_center= (int(batch['X'][0,k, 0].item()),int(batch['X'][0,k, 1].item()))
                print(coor_center)
                cv2.circle(frame, coor_center, 10, (0,255,0) , -1)
                k+=1
            
                size=frame.shape
                frame=cv2.resize(frame, (int(scale* size[1]), int(scale*size[0])))
                cv2.imshow('Title',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_nr+=+1
        cap.release()
        cv2.destroyAllWindows()

    # When everything done, release the capture


    img=plt.imread(img_path)
    print(batch['F'])

    plt.imshow(img)
    plt.plot(np.array(batch['X'][:,:, 0].data.cpu().numpy()[0] ),np.array(batch['X'][:,:, 1].data.cpu().numpy()[0] ))
    plt.show()
    """
