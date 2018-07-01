# load packages and dependencies
import argparse
import os 
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
from NN import model
import time
import logging
import datetime 
from utils.initialize_logger import init_logger
from utils.data import create_dataset, generate_data 


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""This notebook demosntrates the three most common types of recurrent neural networks. Namely, we focus on:
Simple recurrent neural network (RNN) Gated recurrent units (GRU) Long short term memory netowrk (LSTM) 
The models are nicely demonstrated and explained in the following post: 
http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
The models are trained on a one dimensional time series of a noisy sin-wave.""")
parser.add_argument('--train', dest="train", default=False, action='store_true',
                    help='an integer for the accumulator')
parser.add_argument('--test',  dest='test', type=str,
                    default=False,
                    help='sum the integers (default: find the max)')
parser.add_argument('--lb', dest='history_window',  type=int, default=200, help='look back time window') 
parser.add_argument('--lf' , dest='prediction_window', type=int, default =  1, help='prediction time horizont') 
parser.add_argument('--d', dest='data', type=str, default='sine',  help='data set to be used')
parser.add_argument('--hs', dest='hidden_size',type=int,  default=150, help='Number of hidden states')  
parser.add_argument('--nl',dest='number_layer', type=int, default = 3, help='number of RNN layers') 
parser.add_argument('--dr',dest='dropout_rate', type=float, default= 0.3, help='dropout rate for training') 
parser.add_argument('--e', dest='epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--vp', dest='visdom_port',type=int,  default=False, help='port of visdom server')
parser.add_argument('--ft', dest='future',type=int,  default=10, help='future time window')

args = parser.parse_args()
print(args)

if args.visdom_port:
    import visdom
    
    #os.system('python -m visdom.server -port {} &'.format(args.visdom_port))
    vis = visdom.Visdom(port=args.visdom_port)
    vis.close()


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
features= 1 
np.random.seed(0)
look_back=args.history_window        # historic time window
look_forward=args.prediction_window    # prediction time horizont
hidden_size=args.hidden_size # dimension of hidden variable h
num_layer=args.number_layer       # number of LSTM layers
dropout=args.dropout_rate   # dropout rate after each LSTM layer
future = args.future
epochs=args.epochs

sample_size=3000
dataset=generate_data(args.data , sample_size)
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size

if args.visdom_port:
    vis.line(
        Y=dataset,
        X=np.arange(len(dataset)), 
        opts=dict(title="Data set", markers=False),
    )

train_X, train_Y, test_X, test_Y = create_dataset(dataset,train_size,  look_back, look_forward)
# Convert numpy array to PyTorch tensor

train_X = train_X.reshape(look_back, -1, features)
train_Y = train_Y.reshape(features, -1, look_forward)
test_X = test_X.reshape(look_back, -1, features)
test_Y = test_Y.reshape(features, -1, look_forward)

train_x = torch.from_numpy(train_X).cuda()
train_y = torch.from_numpy(train_Y).cuda()
test_x = torch.from_numpy(test_X).cuda()
test_y = torch.from_numpy(test_Y).cuda()

print("Shape train data X: ", train_x.size())
print("Shape train data Y: ", train_y.size())
print("Shape test data X: ", test_x.size())
print("Shape test data Y: " , test_y.size())
# initialize neural nets
init_logger(model_path)

def test(net, models, x=test_x, y=test_y):

    test={}
    
    logging.info('-'* 20 + ' Evaluation Test ' +'-'*20)
    for name in models: 
        net[name].eval()

        test[name] = net[name].predict(x[:,0,0].clone().view(-1, 1, 1), future)
        print("{} future".format(name), criterion(y[0,:future,0].clone().cuda().type('torch.FloatTensor') , torch.from_numpy(test[name].copy()).cuda().type('torch.FloatTensor')).item())
        #loss = criterion(test[name], test_y[:,0,0]) 
        #logging.info('{}: Loss: {:.5f}'.format(name, loss.item()))

    if args.visdom_port:
        win2= vis.line(Y=y[0,:,0].data.cpu().numpy(), 
        
                    X= np.arange(y.size(1)), 
    
        name="real",
        opts=dict(showlegend=True),  
        
        ) 
    #test_y[0].view(-1).data.cpu().numpy()

        for name in models: 
            h0=torch.zeros(net[name].num_layers,x.size(1), net[name].hidden_size).cuda()
        
            data, _ =net[name](x, h0)

            vis.line(Y=data[0, :, 0].data.cpu().numpy(), 
                    X= np.arange(y.size(1)), 
                    name=name, 
                    win=win2,
                    update='new', 
                    opts=dict(showlegend=True), ) 


        win= vis.line(Y=y[0,:future,0].data.cpu().numpy(), 
        X=np.arange(future),
        name="real",
        opts=dict(showlegend=True),  
        
        ) 
    
    #test_y[0].view(-1).data.cpu().numpy()

        for name in models: 
            
            vis.line(Y=test[name], 
                    X=np.arange(future), 
                    name=name, 
                    win=win,
                    update='new', 
                    opts=dict(showlegend=True), ) 


# testing
if args.train:
    batch_size=40
    net={} 
    models=[  'RNN', 'GRU'] #'RNN',,  'GRU' #'LSTM',
    optimizer={} 
    for name in models: 
        net[name]=model(name, features, hidden_size, look_forward, num_layer, dropout, batch_size).cuda() 
        optimizer[name] = torch.optim.Adam(net[name].parameters(), lr=1e-3, weight_decay=1e-4)

    logging.info('Model: {}'.format(models) ) 
    criterion = nn.MSELoss()
    logging.info(net)
    logging.info("Training data set: {}".format(args.data))
    t0= time.time()
    steps= epochs/10

   
    losses={}
    for name in models: 
         losses[name]=[]
    for e in range(epochs):
        
        
        for batch in np.arange(0, train_x.size(1), batch_size):
            var_x = Variable(train_x[:,batch:batch+batch_size,:])
            var_y = Variable(train_y[:,batch:batch+batch_size,:])
            for name in models:
                h0= torch.zeros(net[name].num_layers,var_x.size(1), net[name].hidden_size).cuda()
                out = net[name](var_x, h0)[0]
                
                loss = criterion(out, var_y)
                optimizer[name].zero_grad()
                loss.backward()
                losses[name].append(loss.item())
                optimizer[name].step()    
                if (e+1) % steps == 0: 
                    #logging.info('{}: Loss: {:.5f}'.format(name, loss.item()))
                    """
                    if args.visdom_port:
                        vis.line(
                        X=np.arange(len(losses[name])),
                        Y=np.array(losses[name]),
                        win=name,     
                        opts=dict(
                            ytype='log', 
                            ylabel="MSE error", 
                            xlabel="Epochs", 

                            title="Losses {}".format(name),
                            ), 
                        )
                    """
        dt=time.time()-t0
        t0=dt+t0
        if (e+1)%steps==0:
            for name in models:
                net[name].eval()
                var_x = Variable(train_x)
                var_y = Variable(train_y)
                h0= torch.zeros(net[name].num_layers,var_x.size(1), net[name].hidden_size).cuda()
                
                out = net[name](var_x, h0)[0]
                print(name, criterion(out, var_y).item() )
                net[name].train()
            test(net, models)
            for name in models:
                net[name].train()
             
        logging.info('-'* 20 + ' Epoch {} - {:.2f}% - Time {:.2f}s '.format(e+1, (e+1)/epochs*100, dt) +'-'*20)
    
    
    for name in models:
        torch.save(net[name], os.path.join(model_path, name))
        logging.info("{} saved to {}".format(name, model_path))	

if args.test:
    model_path=os.path.join("models", args.test) 
    net={}
    models=['LSTM', 'RNN', 'GRU']
    for name in models: 
        net[name]=torch.load(os.path.join(model_path, name))
        net[name]=net[name].eval()
        logging.info("{} loaded".format(name))
    criterion= nn.MSELoss() 
# testing 
test(net, models)