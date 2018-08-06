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

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.3, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
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
            hidden = (hx, hx)
        h, c= hidden
        do_dropout = self.training and self.dropout > 0.0
        
        #h = h.view(h.size(1), -1)
        #c = c.view(c.size(1), -1)
        #x = x.view(x.size(1), -1)
        for i in range(input.size(0)):
            x=input[i]
        # Linear mappings
        
            preact = self.i2h(x) + self.h2h(h)
      
            # activations
            gates = preact[:, :3 * self.hidden_size].sigmoid()
            g_t = preact[:, 3 * self.hidden_size:].tanh()
            i_t = gates[:, :self.hidden_size]
            f_t = gates[:, self.hidden_size:2 * self.hidden_size]
            o_t = gates[:, -self.hidden_size:]


            # cell computations
            if do_dropout and self.dropout_method == 'semeniuta':
                g_t = F.dropout(g_t, p=self.dropout, training=self.training)
           
            c = torch.mul(c, f_t) + torch.mul(i_t, g_t)

            if do_dropout and self.dropout_method == 'moon':
                    c.data.set_(torch.mul(c_t, self.mask).data)
                    c.data *= 1.0/(1.0 - self.dropout)

            h = torch.mul(o_t, c.tanh())

            # Reshape for compatibility
            if do_dropout:
                if self.dropout_method == 'pytorch':
                    F.dropout(h, p=self.dropout, training=self.training, inplace=True)
                if self.dropout_method == 'gal':
                        h.data.set_(torch.mul(h, self.mask).data)
                        h.data *= 1.0/(1.0 - self.dropout)

            
            output[i]=h
        return output, (h, c)
