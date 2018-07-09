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
from sacred import Experiment   
import visdom

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

parser.add_argument('--obs',  dest='observe', default=False, action='store_true',
                    help='observe experiement and add to data base')
parser.add_argument('--lb', dest='history_window',  type=int, default=200, help='look back time window') 
parser.add_argument('--lf' , dest='prediction_window', type=int, default =  1, help='prediction time horizont') 
parser.add_argument('--d', dest='data', type=str, default='sine',  help='data set to be used')
parser.add_argument('--hs', dest='hidden_size',type=int,  default=150, help='Number of hidden states')  
parser.add_argument('--nl',dest='number_layer', type=int, default = 3, help='number of RNN layers') 
parser.add_argument('--dr',dest='dropout_rate', type=float, default= 0.5, help='dropout rate for training') 
parser.add_argument('--e', dest='epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--vp', dest='visdom_port',type=int,  default=False, help='port of visdom server')
parser.add_argument('--ft', dest='future',type=int,  default=10, help='future time window')
parser.add_argument('--feat', dest='features',type=int,  default=1, help='number of features')
parser.add_argument('--samp', dest='sample',type=int,  default=2000, help='length of dataset')
parser.add_argument('--ts', dest='t_split',type=float,  default=0.7, help='split training set')
parser.add_argument('--bs', dest='batch_size', type=int, default=12, help='batch_size')




args = parser.parse_args()
print(args)

if args.visdom_port:
    import visdom
    visdom.Visdom(port=args.visdom_port).close()

from sacred.observers import MongoObserver


from sacred.observers import FileStorageObserver

ex = Experiment('RNNs')
if args.observe:
    ex.observers.append(MongoObserver.create( db_name='GPUserver'))
    ex.observers.append(FileStorageObserver.create('scripts'))

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'



@ex.config
def configuration():
    # Parameters
    features= args.features
    seed=0
    look_back=args.history_window        # historic time window
    look_forward=args.prediction_window    # prediction time horizont
    hidden_size=args.hidden_size # dimension of hidden variable h
    num_layer=args.number_layer       # number of LSTM layers
    dropout=args.dropout_rate   # dropout rate after each LSTM layer
    future = args.future
    epochs=args.epochs
    train_bool=args.train
    test_bool=args.test
    sample_size=args.sample
    batch_size=args.batch_size
    t_split=args.t_split
    models=[ 'LSTM', 'GRU', 'RNN'  ] #'RNN',,  'GRU' #'LSTM',
    args=args

@ex.capture
def get_info(_run):
    return  _run.experiment_info["name"], _run._id


@ex.main
def main_func(features,seed,    look_back, look_forward, hidden_size, num_layer, dropout, future , epochs, train_bool, test_bool, args, sample_size,t_split, models,batch_size):
  

    if args.visdom_port:

        vis_env=get_info()
        #os.system('python -m visdom.server -port {} &'.format(args.visdom_port))
        if args.observe:
            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            environment="{}_{}".format(vis_env[0],  vis_env[1]) 
            model_path = os.path.join(model_dir, environment)
            os.makedirs(model_path)
           
            ex.logger= init_logger(model_path)
        else: 
            environment="main"
            ex.logger= init_logger(os.curdir)
        vis = visdom.Visdom(env=environment,  port=args.visdom_port)

    np.random.seed(seed)
    # current directory
    direct= '/'.join(os.getcwd().split('/')[:-1]) 

    start_time       = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    # create dir for model
    """
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, start_time)
    os.makedirs(model_path)
    """
    use_gpu=torch.cuda.is_available()
    print("GPU is available: ",  use_gpu)

    dataset=generate_data(args.data , sample_size)
    train_size = int(len(dataset) * t_split)
    test_size = len(dataset) - train_size
    
    if args.visdom_port:
        
        vis.line(
        X=np.arange(len(dataset)),
        Y=dataset,
        opts=dict(width=1000,
        title="Dataset"),     
        
        )

    train_X, train_Y, test_X, test_Y , val_X, val_Y= create_dataset(dataset,train_size,  look_back, look_forward)
    # Convert numpy array to PyTorch tensor
 
    train_X = train_X.reshape(look_back, -1, features)
    train_Y = train_Y.reshape(look_back, -1, features)
    val_X = val_X.reshape(look_back, -1, features)
    val_Y = val_Y.reshape(look_back, -1, features)
   # test_X = test_X.reshape(look_back, -1, features)
   # test_Y = test_Y.reshape(look_back, -1, features)

    train_x = torch.from_numpy(train_X).cuda()
    train_y = torch.from_numpy(train_Y).cuda()
    val_x = torch.from_numpy(val_X).cuda()
    val_y = torch.from_numpy(val_Y).cuda()
  #  test_x = torch.from_numpy(test_X).cuda()
  #  test_y = torch.from_numpy(test_Y).cuda()

    print("Shape train data X: ", train_x.size())
    print("Shape train data Y: ", train_y.size())
    print("Shape val data X: ", val_x.size())
    print("Shape val data Y: " , val_y.size())
    # initialize neural nets
    
    

    def val(net, models,  ep, x=val_x, y=val_y):
        val={}
        ex.logger.info('-'* 20 + ' Validation ' +'-'*20)
                        
        for name in models: 
            net[name].eval()
            #h0= torch.zeros(net[name].num_layers,x.size(1), net[name].hidden_size).cuda()
            val[name] = net[name](x.clone())
            loss = criterion(val[name], y) 
            ex.logger.info('{}: Val. Loss: {:.5f}'.format(name, loss.item()))
            ex.log_scalar("val_loss_{}".format(name), loss.item(), ep)
            net[name].train()
            if args.visdom_port:

                    vis.line(
                X=np.arange(len(val[name][-1, :,0].data.cpu().numpy())),
                Y=val[name][-1, :,0].data.cpu().numpy(),
                win=val_plot,  
                update='replace',
                name=name,
                opts=dict(showlegend=True), 
                                    )



    def test(net, models, ep, final=False,  x=val_x, y=val_y, args=args):
        T={}
       
        ex.logger.info('-'* 20 + ' Test ' +'-'*20)
        
       
        for name in models: 
            net[name].eval()
            #h0= torch.zeros(net[name].num_layers,x.size(1), net[name].hidden_size).cuda()
            T[name] = net[name].predict(x[:,0,:].clone().unsqueeze(1), y.size(1)-1)
            T_loss=torch.from_numpy(T[name]).cuda()

            loss = criterion(T_loss, y[-1, :, 0]) 
            ex.logger.info('{}: Test. Loss: {:.5f}'.format(name, loss.item()))
            ex.log_scalar("test_loss_{}".format(name), loss.item(), ep)
            if final:
                ex.info["Test {}".format(name)]= loss.item()
                
                ex.info["Std {}".format(name)]= np.std(T[name])

            net[name].train()
            
            if args.visdom_port:
                vis.line(
                X=np.arange(len(T[name]) ),
                Y=T[name],
                win=test_set,  
                update='replace',
                name=name,
                opts=dict(showlegend=True,
                ), 
                                    )
        
        
    # testing
    if train_bool:
        count=0
        net={} 
        if args.visdom_port:
            val_plot=vis.line(
                X=np.arange(len(val_y[-1, :,0])),
                Y=val_y[-1,:,0],
                name="Real",
                opts=dict(
                    
                    title="Validation",
                    showlegend=True,
                    width=700, 
                    ), 
                )
        
            test_set=vis.line(
            X=np.arange(len(val_y[-1 , :,:])),
            Y=val_y[-1, :,:],
            name="Real",
            opts=dict(
                width=1000,
                title="Test prediction",
                ), 
            )
        optimizer={} 
        for name in models: 
            net[name]=model(name, features, hidden_size, look_forward, num_layer, dropout, batch_size).cuda() 
            optimizer[name] = torch.optim.Adam(net[name].parameters(), lr=1e-3, weight_decay=1e-4)
       
        logging.info('Model: {}'.format(models) ) 
        criterion = nn.MSELoss()
        ex.logger.info(net)
        ex.logger.info("Training data set: {}".format(args.data))
        t0= time.time()
        steps= 0.1
        losses={}
        l={}
        for name in models: 
            losses[name]=[]
            
        for e in range(epochs):
            for name in models:
                l[name]=0
            
                for iter, batch in enumerate(np.arange(0, train_x.size(1), batch_size)):
                
                    var_x = Variable(train_x[:,batch:batch+batch_size,:])
                    var_y = Variable(train_y[:,batch:batch+batch_size,:])
                    #h0= torch.zeros(net[name].num_layers,var_x.size(1), net[name].hidden_size).cuda()
    
                    out = net[name](var_x)
                    loss = criterion(out, var_y)
                    optimizer[name].zero_grad()
                    loss.backward()
                    
                    optimizer[name].step()  
                    l[name]+=loss.item()  
                loss_epoch= l[name]/(iter+1)
                losses[name].append(loss_epoch)
                if (e+1)% np.maximum(1, int(epochs*steps))==0: 
            
                    ex.logger.info('{}: Loss: {:.5f}'.format(name,loss_epoch ))
                    ex.log_scalar('loss_{}'.format(name), loss_epoch, e+1)
                
                    if args.visdom_port:
                        vis.line(
                        X=np.arange(len(losses[name])),
                        Y=np.array(losses[name]),
                        win=name,     
                        opts=dict(
                            ytype='log', 
                            ylabel="MSE error", 
                            xlabel="Epochs", 
                            update="new", 
                            title="Losses {}".format(name),
                            ), 
                        )
                        
            dt=time.time()-t0
            t0=dt+t0
            
            
            logging.info('-'* 20 + ' Epoch {} - {:.2f}% - Time {:.2f}s '.format(e+1, (e+1)/epochs*100, dt) +'-'*20)
            if (e+1)% np.maximum(1, int(epochs*steps))==0:
                val(net, models, e+1)
                test(net, models, e+1)
        if args.observe:
            for name in models:
                torch.save(net[name], os.path.join(model_path, name))
                logging.info("{} saved to {}".format(name, model_path))	
        else:
            os.remove("all.log")
    if args.visdom_port and args.observe:
        vis.save(envs=[environment])
    test(net, models,e+1, True)
    
if __name__ == '__main__':
    run=ex.run()
    run.root_logger = None
    run.run_logger=None
  