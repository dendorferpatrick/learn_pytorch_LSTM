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
        self.i2v = nn.Linear(input_size,  3* hidden_size, bias=bias)
        self.v2v = nn.Linear(hidden_size,  3* hidden_size, bias=bias)
        self.a2v=nn.Linear(hidden_size,  hidden_size, bias=bias)
        self.v2a = nn.Linear(hidden_size, 2*  hidden_size, bias=bias)
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

            act_x=self.i2v(x)
            act_v=self.v2v(v)
            act_a=self.a2v(a) 

            add_x=act_x[:, self.hidden_size].tanh()
            g1= (act_x[:, self.hidden_size:2*self.hidden_size]+act_v[:,:self.hidden_size]).sigmoid()
            g2=(act_x[:, 2* self.hidden_size:3*self.hidden_size]+act_v[:,self.hidden_size:2* self.hidden_size]).sigmoid()
            g3=act_v[:,2*self.hidden_size:].sigmoid()

            add_a=act_a.tanh()

        
           

          
            v_new= v+ torch.mul(add_x,g1)+torch.mul(g2, add_a)
            d_v=v_new-v
            act_new_v= self.v2a(d_v)
            g4=act_new_v[:, :self.hidden_size].sigmoid()
            add_new_v=act_new_v[:, self.hidden_size:].tanh()

            a_new=torch.mul(a, g3)+torch.mul(g4, add_new_v)


            # Reshape for compatibility
            if do_dropout:
                if self.dropout_method == 'pytorch':
                    F.dropout(v_new, p=self.dropout, training=self.training, inplace=True)
               

            
            output[i]=v_new
            v=v_new.clone() 
            a=a_new.clone()
        return output, (v_new, a_new)
