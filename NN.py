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
    def __init__(self, module, input_size, hidden_size, output_size, num_layers, dropout, batch_size=1):
        
        
        
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.module=module
        
        
        if self.module=="LSTM":
            super(model, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout) # rnn
           

        elif self.module=="RNN":
            super(model, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout) # rnn
           
        elif self.module=="GRU":
            super(model, self).__init__()
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout) # rnn
          
        else:
            print("No valid model")
        #self.dropout=nn.Dropout(0.5)
        #self.regx = nn.Linear(hidden_size, output_size)
        #self.regy = nn.Linear(hidden_size, output_size)
        self.reg = nn.Linear(hidden_size, output_size)
    
    def set_criterion(self, criterion=nn.MSELoss):
        self.criterion=nn.MSELoss()
    """
    def forward(self, x):
       # print("Input", x.size())
        logger.debug('Input Seq {}'.format(x.size()))
        x = self.rnn(x)[0]
       # print("LSTM", x.size())
        
        logger.debug('RNN {}'.format(x.size()))
        x= self.reg(x)
        logger.debug('NN {}'.format(x.size()))
       # print("LIN out", x.size())
        #out=x.clone() 
        return x
    """
    
    def forward(self, x, v, future=12-1):
        outputs=[]
        logger.debug("Input prediction {}".format(v.size()))
        if self.module=="LSTM":
            v, (h, c)= self.rnn(v) 
            v = self.reg(v)
            logger.debug("Output FC prediction {}".format(v.size()))
            v=v[-1,:, :].unsqueeze(0)
            x+=v
        
            logger.debug("First prediction {}".format(v.size()))
            for i in range(future):# if we should predict the future
                v, (h, c)= self.rnn(v, (h , c))
                v = self.reg(v) 
                
                
                output=torch.cat((output, x+v), 0)
        
        else: 
            x, h= self.rnn(x) 
            x = self.reg(x)
            x=x[-1,:, :].unsqueeze(0)
            output=x
            for i in range(future):# if we should predict the future
                x, h= self.rnn(x, h)
                x = self.reg(x) 
                output=torch.stack((output, x), 0)
        logger.debug(output.size())
       
        return output
