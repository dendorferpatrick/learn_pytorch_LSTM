import sys
from scipy import interpolate
import os
import numpy as np
from sklearn import  linear_model
from pathlib import Path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils
# load test data


import torch 
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Neural netowrk 

class model(nn.Module):
    
    def __init__(self, args,  config):
        super(model, self).__init__()
        self.config=config
        self.module=args.model_name
        self.pred_len=args.pred_len
        self.off=args.off
        self.seq_len=args.seq_len
        
        
    def set_criterion(self, criterion=nn.MSELoss()):
        self.criterion=criterion
        
    
    def predict(self,batch):
        x = batch[0].float().cuda()
        x=x.permute(1, 0, 2)
        x= self.config.scale_x_m(x)

        y = batch[1].float().cuda()
        y=y.permute(1, 0, 2)

        
        for num in np.arange(x.size(1)):
            regr = linear_model.LinearRegression()
          
            regr.fit(np.arange(self.off,self.seq_len).reshape(-1, 1), x[self.off:, num, 0].data.cpu().numpy()) 
            x_pred=regr.predict(np.arange(self.seq_len, (self.seq_len+self.pred_len)).reshape(-1, 1))

            regr = linear_model.LinearRegression()
            regr.fit(np.arange(self.off, self.seq_len).reshape(-1, 1), x[self.off:, num, 1].data.cpu().numpy())
            y_pred=regr.predict(np.arange(self.seq_len, (self.seq_len+self.pred_len)).reshape(-1, 1))

            x_pred=torch.Tensor(x_pred).unsqueeze(1).unsqueeze(2).cuda() 
            y_pred=torch.Tensor(y_pred).unsqueeze(1).unsqueeze(2).cuda()
          
            out=torch.cat((x_pred, y_pred), 2)
            if num==0: 
                output=out
            else:
                output=torch.cat((output, out), 1)
        
        return output, self.config.scale_x_m(y)


