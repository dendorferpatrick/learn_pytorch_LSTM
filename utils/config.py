import numpy as np 
import torch
class config():
    def __init__(self, data):
        self.scale=torch.Tensor(data.scale)
        #self.vis

    def scale_x_m(self, x):
        if x.is_cuda:
            if self.scale.is_cuda:
                pass
            else:
                self.scale=self.scale.cuda()
        else:
            if self.scale.is_cuda:
                self.scale=self.scale.cpu()
        return self.scale[1,:2]+(1+x)/2.*(self.scale[0,:2]-self.scale[1,:2])

    def scale_x_u(self,x):
        if x.is_cuda:
            if self.scale.is_cuda:
                pass
            else:
                self.scale=self.scale.cuda()
        else:
            if self.scale.is_cuda:
                self.scale=self.scale.cpu()
        return -1 + 2*(-self.scale[1,:2]+x)/(self.scale[0,:2]-self.scale[1,:2] )

    def scale_v_m(self, x):
        if x.is_cuda:
            if self.scale.is_cuda:
                pass
            else:
                self.scale=self.scale.cuda()
        else:
            if self.scale.is_cuda:
                self.scale=self.scale.cpu()
        return self.scale[1,2:]+(1+x)/2.*(self.scale[0,2:]-self.scale[1,2:])

    def scale_v_u(self,x):
        if x.is_cuda:
            if self.scale.is_cuda:
                pass
            else:
                self.scale=self.scale.cuda()
        else:
            if self.scale.is_cuda:
                self.scale=self.scale.cpu()
        return -1 + 2*(-self.scale[1,2:]+x)/(self.scale[0,2:]-self.scale[1,2:] )




    