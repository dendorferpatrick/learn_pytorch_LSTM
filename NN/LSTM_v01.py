from torch import nn
import math
import torch

class LSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=00, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2v = nn.Linear(input_size,  hidden_size, bias=bias)
        self.v2v = nn.Linear(hidden_size,  hidden_size, bias=bias)
        self.a2v=nn.Linear(hidden_size,  hidden_size, bias=bias)
        self.v2a = nn.Linear(hidden_size,  hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(torch.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden=None):
        output=torch.empty(input.size(0), input.size(1), self.hidden_size)
        if hidden is None:
            hx = torch.zeros(input.size(1), self.hidden_size,requires_grad=False)
            hidden = (v, a)
        v, a= hidden
        do_dropout = self.training and self.dropout > 0.0
        
        #h = h.view(h.size(1), -1)
        #c = c.view(c.size(1), -1)
        #x = x.view(x.size(1), -1)
        for i in range(input.size(0)):
            x=input[i]
        # Linear mappings
        
            preact = self.i2v(x) + self.v2v(v)
            gates = preact.sigmoid()

            acc_to_vel=self.a2v(a).tanh()
            
            # activations

            f_t = gates[:, self.hidden_size:]
            ag_t=gates[:,: self.hidden_size]
           

          
            v_new= v+ torch.mul(acc_to_vel, ag_t)

            d_v=v_new-v
            new_a_gate= self.v2a(d_v).sigmoid()
            a=torch.mul(a, f_t)+torch.mul(d_v.tanh(), new_a_gate)


            # Reshape for compatibility
            if do_dropout:
                if self.dropout_method == 'pytorch':
                    F.dropout(v_new, p=self.dropout, training=self.training, inplace=True)
               

            
            output[i]=v_new
        return output, (v_new, a)
