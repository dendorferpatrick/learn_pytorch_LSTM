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
import importlib
import time
import logging
import datetime 

import utils 

import glob



def main_func(args):
    

    def checkpoint_path(model_path, epoch):
        return os.path.join(model_path, 'epoch_{}.tar'.format(epoch))
    
    print(args)
    result={}
 
    if args.visdom_port:
        import visdom
        if args.observe:
        
            result["visdom_id"]=args.environment
        else: 
            environment="main"
      
        vis = visdom.Visdom(env=args.environment,  port=args.visdom_port)
        
        #os.system('python -m visdom.server -port {} &'.format(args.visdom_port))
    if args.observe:
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        
        model_path = os.path.join(model_dir, args.environment)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
         
        

    direct= '/'.join(os.getcwd().split('/')[:-1]) 

   
    use_gpu=torch.cuda.is_available()
    print("GPU is available: %s" % use_gpu)



    
 
    train_data = utils.Dataset(args) 
    test_data = utils.Dataset(args, phase="test") 
    logging.debug("Finished loading data")
    config=utils.config(train_data)
    train_loader = DataLoader(train_data.data, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader= DataLoader(test_data.data, batch_size=len(test_data.data), num_workers=4, shuffle=True)
    
    metrics=utils.metrics("average_loss", "final_loss","mean_loss", "loss")
    # testing
    #if train_bool:
    count=0
    
    
    # initialize model
    model=importlib.import_module('NN.{}'.format(args.module)).model
    net=model(args, config).cuda() 
    
    # initializer optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.set_criterion(utils.AVERAGE)

    t0= time.time()
    losses=utils.losses()
    steps=0.05
    logging.debug("Start training")
    for epoch in range(args.epochs):
        net.train() 
        #net.velocity.eval()
        l=0
        for iter, batch in enumerate(train_loader):          
            out, target = net(batch)
            
            loss = net.criterion(out, target)
            
            net.zero_grad()
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()  
            losses.update(loss.item())
            
     
        print('EPOCH {}: {} Loss: {:.5f}'.format(epoch, args.model_name,losses.get() ))
        metrics.update("loss",losses.get(), epoch)
        if epoch %(int(steps*args.epochs)+1)==0:
            average, final, mean= utils.eval(net,  epoch, test_loader, args,  config, "test")
            metrics.update("final_loss",final, epoch)
            metrics.update("average_loss",average, epoch)
            metrics.update("mean_loss",mean, epoch)

            if args.visdom_port:
                metrics.plot("loss", vis)
                metrics.plot("final_loss", vis)
                metrics.plot("mean_loss", vis)
                metrics.plot("average_loss", vis)
        
            #ex.log_scalar('loss_{}'.format(model_type), loss_epoch, e+1)
        
            if args.observe:
                torch.save(net, checkpoint_path(model_path, epoch))   
            
                print("EPOCH {}: {} saved to {}".format(epoch, args.model_name, model_path))	
        
        dt=time.time()-t0
        t0=dt+t0
        
      
    average, final, mean= utils.eval(net,  args.epochs, test_loader, args,  config, "test")
    if args.visdom_port and args.observe:
        vis.save(envs=[args.environment])
    if args.observe:
        torch.save(net, checkpoint_path(model_path, epoch))   
        
        print("EPOCH {}: {} saved to {}".format(epoch, args.model_name, model_path))	
        
    return average, final, mean
    #return result

  
