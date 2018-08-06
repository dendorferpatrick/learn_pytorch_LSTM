#log_scalar# load packages and dependencies
import argparse
import os 
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from NN3 import model
import time
import logging
import datetime 
import visdom
import utils 

import glob


#logging.disable(logging.DEBUG)

def main_func(mo= "models/59_pred_con_v_hs20_nl1/epoch_299.tar"):
    
    model_type="LSTM"
    environment="main"
            #logger= init_logger(os.curdir)
        #environment="main"
    vis = visdom.Visdom(env=environment,  port=8892)


    data_path= "/usr/wiss/dendorfp/dvl/projects/trajnet/data/test_preprocessed"
    look_back= 8 
    look_forward=12
    data = utils.Dataset(look_back, look_forward, val_split=0.01)
    logging.debug("Finished loading data")
    config=utils.config(data)
    


    eval_loader= DataLoader(data.val, batch_size=len(data.val), num_workers=4, shuffle=True)
 
    # initialize model
    net=torch.load(mo).cuda()
    for batch in eval_loader:
        print(batch)
        net.eval()
        x = batch[0].float().cuda()
        x=x.permute(1, 0, 2)
        x=net.config.scale_x_m(x)
       
        val, y = net.predict(batch)
        #logger.debug('validate out {0}, target {1}'.format(val.size(), y.size()))
        #loss_final = net.criterion(val, y) 
        #loss_mean = net.criterion(val[-1], y[-1]) 
        loss_mean=utils.ADE(val, y)
        
        loss_final = utils.FDE(val, y)
        loss_average=utils.AVERAGE(val, y)
        print('{}: Final. Loss: {:.5f}'.format(model_type, loss_final.item()))
        print('{}: Mean. Loss: {:.5f}'.format(model_type, loss_mean.item()))
        print('{}: AVERAGE.  Loss: {:.5f}'.format(model_type, loss_average.item()))

        #ex.log_scalar("val_loss_{}".format(model_type), loss.item(), ep)
        for i in np.arange(val.size(1))[:10]:
            vis.line(
                #X=val[:,i, 0 ].unsqueeze(1),
                #Y=val[:,i, 1 ].unsqueeze(1), 
            X=torch.cat((x[:,i, 0 ].unsqueeze(1),      val[:,i, 0 ].unsqueeze(1))  ), 
            Y=torch.cat((x[:,i, 1 ].unsqueeze(1),     val[:,i, 1 ].unsqueeze(1))  ), 
            #update='replace',
            name="test",
            #win="test",
            opts=dict(showlegend=True, 
                xtickmin=config.scale[1, 0].item(),
                xtickmax=config.scale[0, 0].item(),
                ytickmin=config.scale[1, 1].item(),
                ytickmax=config.scale[0, 1].item(),
                width=500, 
                height=500,
               # legend=['Prediction', 'Ground truth'], 
                xlabel= 'x', 
                ylabel= 'y',) ,
                                )
            
if __name__=="__main__":
    main_func()
