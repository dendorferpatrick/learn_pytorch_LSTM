import torch 
from torch import nn 
from torch.autograd import Variable
import numpy as np
# Neural netowrk 

class model(nn.Module):
    def __init__(self, module, input_size, hidden_size, output_size, num_layers, dropout, batch_size=1):
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        
        
        if module=="LSTM":
            super(model, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout) # rnn
           

        elif module=="RNN":
            super(model, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout) # rnn
           
        elif module=="GRU":
            super(model, self).__init__()
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout) # rnn
          
        else:
            print("No valid model")
        self.dropout=nn.Dropout(0.5)
        self.reg = nn.Linear(hidden_size, output_size)
        

    def forward(self, x, h):
        
        #h0 = torch.normal(torch.zeros(self.num_layers, x.size(1), self.hidden_size), torch.ones(self.num_layers, x.size(1), self.hidden_size) *.7).cuda()
        #c0 = torch.normal(torch.zeros(self.num_layers, x.size(1), self.hidden_size), torch.ones(self.num_layers, x.size(1), self.hidden_size) *.7).cuda()

        #c0 = - torch.ones(self.num_layers, x.size(1), self.hidden_size).cuda()
        out, h = self.rnn(x, h) # (seq, batch, hidden)
        print(h.size())
        h=out[1, :, :].unsqueeze(0)
        out=self.dropout(out)
        out=out[-1, :,:]
        
        out = self.reg(out)
        out=out.unsqueeze(0)
        return out, h

    def predict(self, x, future=0):
        outputs=[]
      
        h_0= torch.zeros(self.num_layers, 1, self.hidden_size).cuda()
        for i in range(future):# if we should predict the future
            out, h_0= self.forward(x, h_0 )
            
            
            x[:-1, :,:]=x[1:, :, :]
          
            x[-1, :, :]= out[-1, :,:].item()

            outputs.append(out[-1,:,:].item())

       
        return np.array(outputs)
