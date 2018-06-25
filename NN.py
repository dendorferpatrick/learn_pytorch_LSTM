import torch 
from torch import nn 
from torch.autograd import Variable
# Neural netowrk 

class model(nn.Module):
    def __init__(self, module, input_size, hidden_size, output_size, num_layers, dropout):
        if module=="LSTM":
            super(model, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout) # rnn
            self.reg = nn.Linear(hidden_size, output_size)

        elif module=="RNN":
            super(model, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout) # rnn
            self.reg = nn.Linear(hidden_size, output_size)
        elif module=="GRU":
            super(model, self).__init__()
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout) # rnn
            self.reg = nn.Linear(hidden_size, output_size)
        else:
            print("No valid model")

    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

    def predict(self, inp, input_size, future=0):
        outputs=[]
        for i in range(future):# if we should predict the future
            x, _ = self.rnn(inp) # (seq, batch, hidden)
            s, b, h = x.shape
            x = x.view(s*b, h)
            x = self.reg(x)
            x = x.view(s, b, -1)
            outputs += [x]
            inp[:,:,:(input_size-1)]=inp[:,:,1:]
            inp[:,:,-1]=x[-1].item()
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
