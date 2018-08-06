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
from main import main_func
import socket


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
parser.add_argument('--lb', dest='history_window',  type=int, default=8, help='look back time window') 
parser.add_argument('--lf' , dest='prediction_window', type=int, default = 1, help='prediction time horizont') 
parser.add_argument('--d', dest='data', type=str, default='sine',  help='data set to be used')
parser.add_argument('--hs', dest='hidden_size',type=int,  default=20, help='Number of hidden states')  
parser.add_argument('--nl',dest='number_layer', type=int, default = 2, help='number of RNN layers') 
parser.add_argument('--dr',dest='dropout_rate', type=float, default= 0.5, help='dropout rate for training') 
parser.add_argument('--e', dest='epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--vp', dest='visdom_port',type=int,  default=False, help='port of visdom server')
parser.add_argument('--ft', dest='future',type=int,  default=10, help='future time window')
parser.add_argument('--feat', dest='features',type=int,  default=2, help='number of features')
parser.add_argument('--samp', dest='sample',type=int,  default=2000, help='length of dataset')
parser.add_argument('--ts', dest='t_split',type=float,  default=0.8, help='split training set')
parser.add_argument('--bs', dest='batch_size', type=int, default=12, help='batch_size')
parser.add_argument('--m', dest='model', type=str, default="LSTM", help='RNN model')
parser.add_argument('--mode', dest='train_mode', type=str, default="m2m", help='train model (m2m, m2o)')






args = parser.parse_args()
print(args)


from sacred.observers import MongoObserver


from sacred.observers import FileStorageObserver
name_ex="Test"
ex = Experiment(name_ex)
if args.observe:
    ex.observers.append(MongoObserver.create( url = "mongodb://%s:%s@%s/%s" % ("dendorfp", "mongo", "hpccremers4", "traj_RNN") ))
    

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
    model_type=args.model #'RNN',,  'GRU' #'LSTM',
    args=args
    train_mode=args.train_mode
@ex.capture
def get_info(_run):
    return  _run.experiment_info["name"], _run._id

def write_config(_run, dic):
    return  ex.add_config(dic)


@ex.main
def run_main(features,seed,    look_back, look_forward, hidden_size, num_layer, dropout, future , epochs, train_bool, test_bool, args, sample_size,t_split, model_type,batch_size, train_mode):
    print("Test")
if __name__ == '__main__':
    run=ex.run()
  
    #run.root_logger = None
    #run.run_logger
  