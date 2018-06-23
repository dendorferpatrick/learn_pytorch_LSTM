print("This notebook demosntrates the three most common types of recurrent neural networks. Namely, we focus on:\n", 
        "Simple recurrent neural network (RNN) Gated recurrent units (GRU) Long short term memory netowrk (LSTM) \n \n", 
        "The models are nicely demonstrated and explained in the following post: \n ", 
        "http://colah.github.io/posts/2015-08-Understanding-LSTMs/ \n" , 
        "The models are trained on a one dimensional time series of a noisy sin-wave.")

# load packages and dependencies
import os 
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
import time
import logging
import datetime 
def initialize_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
     
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s -%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "error.log"),"w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s -%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "all.log"),"w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s -%(levelname)s - %(message)s")    
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# current directory
direct= '/'.join(os.getcwd().split('/')[:-1]) 

start_time       = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, start_time)
os.makedirs(model_path)

use_gpu=torch.cuda.is_available()
print("GPU is available: ",  use_gpu)

# Parameters
np.random.seed(0)
look_back=200        # historic time window
look_forward=100    # prediction time horizont
hidden_size=130       # dimension of hidden variable h
num_layer=3          # number of LSTM layers
dropout=   0.3      # dropout rate after each LSTM layer

epochs=1000

sample_size=3000

# generate data (noisy sin waves)

def sine_2(X, signal_freq=60.):

    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def sample(sample_size=sample_size):
    random_offset = np.random.randint(0, sample_size)
    X = np.arange(sample_size)
    Y = noisy(sine_2(X + random_offset)).astype('float32')
    return Y


dataset=sample()
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size

# Normalize data to [0, 1]

max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))

def create_dataset(dataset, look_back, look_forward):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back- look_forward):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back:(i + look_back+look_forward)])
    return np.array(dataX), np.array(dataY)


data_X, data_Y = create_dataset(dataset, look_back, look_forward)

# Slit data to train and test data
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

# Convert numpy array to PyTorch tensor
train_X = train_X.reshape(-1, 1, look_back)
train_Y = train_Y.reshape(-1, 1, look_forward)
test_X = test_X.reshape(-1, 1, look_back)
test_Y = test_Y.reshape(-1, 1, look_forward)

train_x = torch.from_numpy(train_X).cuda()
train_y = torch.from_numpy(train_Y).cuda()
test_x = torch.from_numpy(test_X).cuda()
test_y = torch.from_numpy(test_Y).cuda()

print(train_x.size())
print(train_y.size())
print(test_x.size())
print(test_y.size())

# Neural netowrk 

class model(nn.Module):
    def __init__(self, module, input_size, hidden_size, output_size, num_layers=num_layer, dropout=dropout):
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

# initialize neural nets
initialize_logger(model_path)
net={}
models=['LSTM', 'RNN', 'GRU']
optimizer={}
for name in models:
    net[name]=model(name, look_back, hidden_size, look_forward).cuda()
    optimizer[name] = torch.optim.Adam(net[name].parameters(), lr=1e-2)
logging.info('Model: {}'.format(models) ) 
criterion = nn.MSELoss()
logging.info(net)
#logging.basicConfig(filename=os.path.join(model_path, start_time+'.log') ,  format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level= logging.DEBUG)

# training 
t0= time.time()
for e in range(epochs):
    if (e+1) % 100 == 0: 
        dt=time.time()-t0
        t0=dt+t0
        logging.info('-'* 20 + ' Epoch {} - {:.2f}% - Time {:.2f}s '.format(e+1, (e+1)/epochs*100, dt) +'-'*20)
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    for name in models:
        out = net[name](var_x)
        loss = criterion(out, var_y)
        optimizer[name].zero_grad()
        loss.backward()
        optimizer[name].step()
        if (e+1) % 100 == 0: 
            logging.info('{}: Loss: {:.5f}'.format(name, loss.item()))
           
for name in models:
    net[name] = net[name].eval() 

test={}
test_x=Variable(test_x)
logging.info('-'* 20 + ' Evaluation Test ' +'-'*20)
for name in models: 
    test[name] = net[name](test_x)
    loss = criterion(test[name], test_y)
    torch.save(name[name], os.path.join(model_path, name) )
    logging.info('{}: Loss: {:.5f}'.format(name, loss.item()))

