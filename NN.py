import torch 
from torch import nn 
from torch.autograd import Variable
import numpy as np
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
        self.dropout=nn.Dropout(0.5)
        self.reg = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):

        out = self.rnn(x)[0] # (seq, batch, hidden)
       
        out=self.dropout(out)
        out = self.reg(out)
        return out

    def predict(self, x, future=0):
        outputs=[]
        if self.module=="LSTM":
            c_0=torch.zeros(self.num_layers, 1, self.hidden_size).cuda()
            h_0= torch.zeros(self.num_layers, 1, self.hidden_size).cuda()
            out, (h, c)= self.rnn(x, (h_0 , c_0)) 
            out = self.reg(out)
            out=out[-1,:, :].unsqueeze(0)
            outputs.append(out[0,0,0].item())     
            for i in range(future):# if we should predict the future
                out, (h, c)= self.rnn(out, (h , c))
                out = self.reg(out) 
                outputs.append(out[0,0,0].item())
        
        else: 
            h_0= torch.zeros(self.num_layers, 1, self.hidden_size).cuda()
            out, h= self.rnn(x, h_0 )
            out = self.reg(out)
            out=out[-1,:, :].unsqueeze(0)
            outputs.append(out[0,0,0].item())     
            for i in range(future):# if we should predict the future
                out, h= self.rnn(out, h )
                out = self.reg(out) 
                outputs.append(out[0,0,0].item())

       
        return np.array(outputs).astype('float32')
