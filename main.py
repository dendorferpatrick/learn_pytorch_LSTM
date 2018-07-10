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

import visdom

def main_func(features,seed,    look_back, look_forward, hidden_size, num_layer, dropout, future , epochs, train_bool, test_bool, args, sample_size,t_split, model_type,batch_size, train_mode, vis_env):
    print(args)
    result={}
    if args.visdom_port:
        
        #os.system('python -m visdom.server -port {} &'.format(args.visdom_port))
        if args.observe:
            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            environment="{}_{}".format(vis_env[0],  vis_env[1]) 
            result["visdom_id"]=environment
            model_path = os.path.join(model_dir, environment)
            os.makedirs(model_path)
            #init_logger(model_path)
        else: 
            environment="main"
            #logger= init_logger(os.curdir)
        vis = visdom.Visdom(env=environment,  port=args.visdom_port)

    np.random.seed(seed)
    # current directory
    direct= '/'.join(os.getcwd().split('/')[:-1]) 

    start_time       = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    # create dir for model
   
    use_gpu=torch.cuda.is_available()
    print("GPU is available: %s" % use_gpu)

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

    print("Shape train data X: " + str(train_x.size()) )
    print("Shape train data Y: " + str(train_y.size()))
    print("Shape val data X: " + str(val_x.size()))
    print("Shape val data Y: " + str(val_y.size()))
    # initialize neural nets
    
    

    def val(net, model_type,  ep, x=val_x, y=val_y):
    
        print('-'* 20 + ' Validation ' +'-'*20)
                        
       
        net.eval()
        #h0= torch.zeros(net.num_layers,x.size(1), net.hidden_size).cuda()
        val = net(x.clone())
        loss = criterion(val, y) 
        print('{}: Val. Loss: {:.5f}'.format(model_type, loss.item()))
        #ex.log_scalar("val_loss_{}".format(model_type), loss.item(), ep)
        net.train()
        result["Val".format(model_type)]= loss.item()
        if args.visdom_port:

                vis.line(
            X=np.arange(len(val[-1, :,0].data.cpu().numpy())),
            Y=val[-1, :,0].data.cpu().numpy(),
            win=val_plot,  
            update='replace',
            name=model_type,
            opts=dict(showlegend=True), 
                                )



    def test(net, model_type, ep, final=False,  x=val_x, y=val_y, args=args):
        T={}
       
        print('-'* 20 + ' Test ' +'-'*20)
        
       
        
        net.eval()
        #h0= torch.zeros(net.num_layers,x.size(1), net.hidden_size).cuda()
        T = net.predict(x[:,0,:].clone().unsqueeze(1), y.size(1)-1)
        T_loss=torch.from_numpy(T).cuda()

        loss = criterion(T_loss, y[-1, :, 0]) 
        print('{}: Test. Loss: {:.5f}'.format(model_type, loss.item()))
        #ex.log_scalar("test_loss_{}".format(model_type), loss.item(), ep)
        if final:
            result["Test".format(model_type)]= loss.item()
            result["STD".format(model_type)]= np.std(T)

        net.train()
        
        if args.visdom_port:
            vis.line(
            X=np.arange(len(T) ),
            Y=T,
            win=test_set,  
            update='replace',
            name=model_type,
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
                    width=1000, 
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
       
        net=model(model_type, features, hidden_size, look_forward, num_layer, dropout, batch_size).cuda() 
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
        print('Model: {}'.format(model_type) ) 
        criterion = nn.MSELoss()
        print(net)
        print("Training data set: {}".format(args.data))
        t0= time.time()
        steps= 0.1
        losses=[]
        l=0
        
        for e in range(epochs):
           
            l=0
        
            for iter, batch in enumerate(np.arange(0, train_x.size(1), batch_size)):
            
                var_x = Variable(train_x[:,batch:batch+batch_size,:])
                var_y = Variable(train_y[:,batch:batch+batch_size,:])
                #h0= torch.zeros(net.num_layers,var_x.size(1), net.hidden_size).cuda()

                out = net(var_x)
                if train_mode=="m2m":
                    loss = criterion(out, var_y)
                elif train_mode=="m2o":
                    loss = criterion(out[-1], var_y[-1])
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()  
                l+=loss.item()  
            loss_epoch= l/(iter+1)
            losses.append(loss_epoch)
            if (e+1)% np.maximum(1, int(epochs*steps))==0: 
        
                print('{}: Loss: {:.5f}'.format(model_type,loss_epoch ))
                #ex.log_scalar('loss_{}'.format(model_type), loss_epoch, e+1)
            
                if args.visdom_port:
                    vis.line(
                    X=np.arange(len(losses)),
                    Y=np.array(losses),
                    win=model_type,     
                    opts=dict(
                        ytype='log', 
                        ylabel="MSE error", 
                        xlabel="Epochs", 
                        update="new", 
                        title="Losses {}".format(model_type),
                        ), 
                    )
                    
            dt=time.time()-t0
            t0=dt+t0
            
            
            print('-'* 20 + ' Epoch {} - {:.2f}% - Time {:.2f}s '.format(e+1, (e+1)/epochs*100, dt) +'-'*20)
          
        if args.observe:
            
            torch.save(net, os.path.join(model_path, model_type))
            print("{} saved to {}".format(model_type, model_path))	
     
    if args.visdom_port and args.observe:
        vis.save(envs=[environment])
    val(net, model_type, e+1)
    test(net, model_type,e+1, True)
    return result

  