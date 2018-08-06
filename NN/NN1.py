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
        self.num_layers=args.number_layer
        self.hidden_size=args.hidden_size
        self.module=args.model_name
        self.dropout_rate=args.dropout_rate
        self.pred_len=args.pred_len


        self.velocity_LSTM=  nn.LSTM(args.features, args.hidden_size, self.num_layers) # rnn
        self.velocity_linear= nn.Linear(self.hidden_size, args.features)
        self.velocity_tanh=nn.Tanh()
        
        self.acceleration_LSTM=  nn.LSTM(args.features, args.hidden_size, self.num_layers) # rnn
        self.acceleration_linear= nn.Linear(self.hidden_size, args.features)
        self.acceleration_tanh=nn.Tanh()
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
        a=v[1:]-v[:-1]
       
        
        v, _= self.velocity_LSTM(v) 
        v = self.velocity_linear(v)
        v=self.velocity_tanh(v)
        v=v[-1,:, :].unsqueeze(0)
        v=self.config.scale_v_m(v)
        v_size=v.size() 
        logger.debug("velocity {}".format( v_size))

        a, (h, c)= self.acceleration_LSTM(a) 
        a = self.acceleration_linear(a)
        a=self.acceleration_tanh(a)
        a=a[-1,:, :].unsqueeze(0)
        a_m=self.config.scale_v_m(a)
        a_size=a.size() 
        logger.debug("accleration  {}".format(a_size))
        x_size=x.size() 
        logger.debug("x  {}" .format(x_size))
        v+=a_m
        x+=v
        
        output=x
        
        for i in range(self.pred_len-1):# if we should predict the future
            a, (h, c)= self.acceleration_LSTM(a) 
            a = self.acceleration_linear(a)
            a=self.acceleration_tanh(a)
            a_m=self.config.scale_v_m(a)
            v+=a_m
            x+=v
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
        a=v[1:]-v[:-1]
       
        
        v, _= self.velocity_LSTM(v) 
        v = self.velocity_linear(v)
        v=self.velocity_tanh(v)
        v=v[-1,:, :].unsqueeze(0)
        v=self.config.scale_v_m(v)

        a, (h, c)= self.acceleration_LSTM(a) 
        a = self.acceleration_linear(a)
        a=self.acceleration_tanh(a)
        a=a[-1,:, :].unsqueeze(0)
        a_m=self.config.scale_v_m(a)
        
        v+=a_m
        x+=v
        output=x
        
        for i in range(self.pred_len-1):# if we should predict the future
            a, (h, c)= self.acceleration_LSTM(a) 
            a = self.acceleration_linear(a)
            a=self.acceleration_tanh(a)
            a_m=self.config.scale_v_m(a)
            v+=a_m
            x+=v
            output=torch.cat((output, x), 0)
        
       
        logger.debug(output.size())
        return output, self.config.scale_x_m(y)