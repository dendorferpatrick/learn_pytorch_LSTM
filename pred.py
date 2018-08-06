import sys
from scipy import interpolate
import os
import numpy as np
from sklearn import  linear_model
from pathlib import Path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils_traj as ut
from torch.utils.data import DataLoader
import torch
import pandas as pd
import utils
import logging
import visdom
import glob
# load test data

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.disable(logging.DEBUG)
def get_txt_files(base_dir):
    return glob.iglob(f"{base_dir}/**/*.txt", recursive=True)


def test(model,epoch, off=0): 
    print(model.split('/'))
    xmax=np.array([36.737 ,   63.559])
    xmin=np.array([-37.029, -28.242])
    path_model=os.path.join("models", model, "epoch_{}.tar".format(epoch))
   
    net=torch.load(path_model).cuda()
    print(net.config.scale)
    net.eval()
    print("Network loaded")
    cur_direct= os.getcwd()
    off=off
    path_data=os.path.join(str(Path(__file__).resolve().parents[1]), 'data/test_preprocessed')
    path_out=os.path.join(os.getcwd(),'output/{}'.format(model))
    
    if os.path.exists(path_out)==False:
            os.mkdir(path_out)
    FOLDERS=["stanford", "biwi", "crowds"]
    for folder in FOLDERS:
        path_folder=os.path.join(path_out, folder)
        if os.path.exists(path_folder)==False:
            os.mkdir(path_folder)
    os.chdir(path_data)
    data=get_txt_files(".")

    for file in data: 
        print(file)
        test_data=np.genfromtxt(file, delimiter=" ") 
        steps=len(test_data)
        for s in np.arange(0,steps, 20 ):
            x=torch.Tensor(test_data[s:s+8, 2:4]).unsqueeze(0).float().cuda()
            v=x[:,1:]-x[:, :-1]
            x= net.config.scale_x_u(x.permute(1,0,2)).permute(1,0,2)
          
            v= net.config.scale_v_u(v.permute(1,0,2)).permute(1,0,2)
            #print(v)
            batch=[x,x,v,v]
            output, _=net.predict(batch)
            x=net.config.scale_x_m(x.permute(1,0,2))
            #print(output[1:, 0, :].data.cpu().numpy()- output[:-1, 0, :].data.cpu().numpy())
            test_data[s+8:s+20, 2:4]=output[:, 0, :].data.cpu().numpy()
       
        path=os.path.join(path_out, file.split('/')[-2], file.split('/')[-1])
        np.savetxt(path, test_data, delimiter=" ",  fmt=('%i', '%.1f', '%.3f', '%.3f')  )
        logger.info("Saved: %s" %path) 

    os.chdir(path_out)
   
    zipfile="{}_{}.zip".format(*model.split('/') )
    file =os.path.join(path_out, zipfile)
    print(file)
    os.system("zip -r %s . -x '*.DS_Store'" %file)
    sess=ut.Session()
    sess.start() 
    sess.login()
    sess.submit( model, file)
    #os.chdir(cur_direct)
    print(sess.get_score(model))

    
if __name__== '__main__':
    test("hpccremers4/96_LSTM_v02", 399)
