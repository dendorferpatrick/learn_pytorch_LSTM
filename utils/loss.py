import torch

def ADE(x, y): 
    num=x.size(1)
    pred_length= x.size(0)
    error=torch.sum(torch.norm(x-y,2, 2)) /(pred_length*num)
    return error
def FDE(x, y):
    num=x.size(1)
   
    error= torch.sum(torch.norm(x[-1]-y[-1], 2, 1))/num
    return error
def AVERAGE(x, y):
    error= (ADE(x, y)+ FDE(x, y))/2.
    return error