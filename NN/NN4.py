import torch 
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import logging
from NN.LSTM_v02 import LSTM 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Neural netowrk 

class model(nn.Module):
    
    def __init__(self, args,  config):
        super(model, self).__init__()
        self.config=config
        self.num_layers=args.number_layer
        self.hidden_size=args.hidden_size
        self.module=args.model_name
        self.dropout_rate=args.dropout_rate
        self.pred_len=args.pred_len
        
      
        
        if self.num_layers>1:
            self.rnn = nn.LSTM(args.features, args.hidden_size, self.num_layers, dropout=args.dropout_rate) # rnn
        else: 
            self.rnn = nn.LSTM(args.features, args.hidden_size) # rnn
           
        self.reg = nn.Linear(self.hidden_size, args.features)
    
    def set_criterion(self, criterion=nn.MSELoss()):
        self.criterion=criterion
        
    def forward(self, batch):
        x = batch[0].float().cuda()
        x=x.permute(1, 0, 2)
        x=x[-1].unsqueeze(0)
        x= self.config.scale_x_m(x)
        y = batch[1].float().cuda()
        y=y.permute(1, 0, 2)
        v = batch[2].float().cuda()
        v=v.permute(1, 0, 2)
       
        logger.debug("Input prediction {}".format(v.size()))
       
        v, (h, c)= self.rnn(v) 
        v = self.reg(v)
        logger.debug("Output FC prediction {}".format(v.size()))
        v=v[-1,:, :].unsqueeze(0)
        
        x+=self.config.scale_v_m(v)
        output=x
        logger.debug("First prediction {}".format(v.size()))
        for i in range(self.pred_len-1):# if we should predict the future
            v, (h, c)= self.rnn(v, (h , c))
            v = self.reg(v) 

            x+=self.config.scale_v_m(v)
            output=torch.cat((output, x), 0)
        
     
        logger.debug(output.size())
    
        return output, self.config.scale_x_m(y)


    def predict(self,batch):
        

        x = batch[0].float().cuda()
        x=x.permute(1, 0, 2)
        x=x[-1].unsqueeze(0)
        x= self.config.scale_x_m(x)
        y = batch[1].float().cuda()
        y=y.permute(1, 0, 2)
        v = batch[2].float().cuda()
        v=v.permute(1, 0, 2)
       
        logger.debug("Input prediction {}".format(v.size()))
       
        v, (h, c)= self.rnn(v) 
        v = self.reg(v)
        logger.debug("Output FC prediction {}".format(v.size()))
        v=v[-1,:, :].unsqueeze(0)
        
        x+=self.config.scale_v_m(v)
        output=x
        logger.debug("First prediction {}".format(v.size()))
        for i in range(self.pred_len-1):# if we should predict the future
            v, (h, c)= self.rnn(v, (h , c))
            v = self.reg(v) 

            x+=self.config.scale_v_m(v)
            output=torch.cat((output, x), 0)
        
     
        logger.debug(output.size())
    
        return output, self.config.scale_x_m(y)
