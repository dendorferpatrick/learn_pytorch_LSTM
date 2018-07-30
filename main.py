#log_scalar# load packages and dependencies
import argparse
import os 
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from NN3 import model
import time
import logging
import datetime 

import utils 

import glob


logging.disable(logging.DEBUG)

def main_func(features,seed,    look_back, look_forward, hidden_size, num_layer, dropout, future , epochs, train_bool, test_bool, args, sample_size,t_split, model_type,batch_size, train_mode, vis_env):
    
    def checkpoint_path(model_path, epoch):
        return os.path.join(model_path, 'epoch_{}.tar'.format(epoch))
    
    print(args)
    result={}
    if args.visdom_port:
        import visdom
        
        #os.system('python -m visdom.server -port {} &'.format(args.visdom_port))
        if args.observe:
            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            environment="{}_{}".format(vis_env[1],  vis_env[0]) 
            result["visdom_id"]=environment
            model_path = os.path.join(model_dir, environment)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            #init_logger(model_path)
        else: 
            environment="main"
            #logger= init_logger(os.curdir)
        #environment="main"
        vis = visdom.Visdom(env=environment,  port=args.visdom_port)

    np.random.seed(seed)
    # current directory
    direct= '/'.join(os.getcwd().split('/')[:-1]) 

    start_time       = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"

    # create dir for model
   
    use_gpu=torch.cuda.is_available()
    print("GPU is available: %s" % use_gpu)



    data_path= "/usr/wiss/dendorfp/dvl/projects/trajnet/data/train"
 
    data = utils.Dataset(look_back, look_forward, val_split=t_split)
    logging.debug("Finished loading data")
    config=utils.config(data)
    train_loader = DataLoader(data.train, batch_size=batch_size, num_workers=4, shuffle=True)
    eval_loader= DataLoader(data.val, batch_size=len(data.val), num_workers=4, shuffle=True)
    
    metrics=utils.metrics("final_loss","mean_loss", "loss")
    # testing
    #if train_bool:
    count=0
    
    
    # initialize model
    net=model(model_type, features, hidden_size, features, num_layer, dropout, batch_size).cuda() 
    
    # initializer optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    net.set_criterion(nn.MSELoss)
    
    #if train_mode=="m2m":
              #  loss = criterion(out, var_y)
           # elif train_mode=="m2o":
               # loss = criterion(out[-1], var_y[-1])
    
    t0= time.time()
    losses=utils.losses()
    steps=0.1
    logging.debug("Start training")
    for epoch in range(epochs):
        l=0
        for iter, batch in enumerate(train_loader):
            
           # var_x = Variable(batch[0]).float().cuda()
           # var_x=var_x.permute(1, 0, 2)
            #var_y = Variable(batch[1]).float().cuda()
            #var_y=var_y.permute(1, 0, 2)
            var_vx = Variable(batch[2]).float().cuda()
            var_vx=var_vx.permute(1, 0, 2)
            var_vy = Variable(batch[3]).float().cuda()
            var_vy=var_vy.permute(1, 0, 2)
            
            out = net( var_vx)
            
            loss = net.criterion(out, var_vy)

            net.zero_grad()
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()  
            losses.update(loss.item())
            
        #loss_epoch= l/(iter+1)
        #losses.append(loss_epoch)
        #if (epoch+1)% np.maximum(1, int(epochs*steps))==0: 
      
        metrics.update("loss",losses.get(), epoch)
        metrics.plot("loss", vis)
        final, mean= utils.validate(net, model_type,  epoch, eval_loader, args, vis, look_back, config)
        metrics.update("final_loss",final, epoch)
        metrics.plot("final_loss", vis)
        metrics.update("mean_loss",mean, epoch)
        metrics.plot("mean_loss", vis)
        print('{}: Loss: {:.5f}'.format(model_type,losses.get() ))
            #ex.log_scalar('loss_{}'.format(model_type), loss_epoch, e+1)
        
        if args.observe:
            torch.save(net, checkpoint_path(model_path, epoch))   
            """
            torch.save({
                'epoch': epochs, 
                'state_dict': net.state_dict(), 
                'optimizer_state_dict':optimizer.state_dict()}, 
                checkpoint_path(model_path, epoch))    
            """
            print("{} saved to {}".format(model_type, model_path))	
        dt=time.time()-t0
        t0=dt+t0
        print('-'* 20 + ' Epoch {} - {:.2f}% - Time {:.2f}s '.format(epoch+1, (epoch+1)/epochs*100, dt) +'-'*20)
        
    
        
        #torch.save(net, os.path.join(model_path, model_type))
        
    

    #val(net, model_type, e+1)
    #test(net, model_type,e+1, True)
    if args.visdom_port and args.observe:
        vis.save(envs=[environment])
   
    #return result

  