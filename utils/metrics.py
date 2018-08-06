import numpy as np

class metrics():
    def __init__(self, *argv):
        self.log={}
        for metric in argv:
            self.log[metric]={"val":[], "epoch": []}

    def update(self, metric, val, epoch):
        self.log[metric]["val"].append(val)
        self.log[metric]["epoch"].append(epoch)
    def add(self, metric):
        self.log[metric]={"val":[], "epoch": []}

    def plot(self, metric, vis): 
        
        vis.line(
        
        X=np.array(self.log[metric]["epoch"]),
        Y=np.array(self.log[metric]["val"]),
        win=metric,  
        #update='replace',
        name=metric,
        opts=dict(showlegend=True, 
            xlabel= 'epoch', 
            ylabel= metric,) ,
                            )

class losses():
    def __init__(self, w=0.6):
        self.w=0.95
        self.loss=[] 
    def update(self, new_loss):
        if len(self.loss)==0: 
            self.loss=[new_loss]
        else:
            self.loss.append(self.loss[-1]*self.w + (1-self.w)* new_loss)
    def get(self):
        return self.loss[-1]