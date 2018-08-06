# load packages and dependencies
import argparse
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
from utils.initialize_logger import init_logger
from utils.data import create_dataset, generate_data 
from sacred import Experiment   
from main import main_func
import socket


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('--obs',  dest='observe', default=False, action='store_true',
                    help='observe experiement and add to data base')
parser.add_argument('--seq', dest='seq_len',  type=int, default=8, help='look back time window') 
parser.add_argument('--pred' , dest='pred_len', type=int, default = 12, help='prediction time horizont') 
parser.add_argument('--hs', dest='hidden_size',type=int,  default=20, help='Number of hidden states')  
parser.add_argument('--nl',dest='number_layer', type=int, default = 2, help='number of RNN layers') 
parser.add_argument('--dr',dest='dropout_rate', type=float, default= 0.5, help='dropout rate for training') 
parser.add_argument('--e', dest='epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--vp', dest='visdom_port',type=int,  default=False, help='port of visdom server')
parser.add_argument('--feat', dest='features',type=int,  default=2, help='number of features')
parser.add_argument('--vs', dest='v_split',type=float,  default=0.2, help='validation split')
parser.add_argument('--ts', dest='t_split',type=float,  default=0.2, help='test split')
parser.add_argument('--bs', dest='batch_size', type=int, default=12, help='batch_size')
parser.add_argument('--m', dest='model_name', type=str, default="unknown", help='name of model')
parser.add_argument('--p', dest='pretrained', type=str, default="unknown", help='name of model')
parser.add_argument('--module', dest='module', type=str, default="unknown", help='name of nn module')
parser.add_argument('--debug', dest="debug", default=False, action='store_true',
                    help='debugging mode')

parser.add_argument('--off', dest='off', type=int, default=1, help='offset_linear')



args = parser.parse_args()
print(args)

if not args.debug:
    logging.disable(logging.DEBUG)

if args.visdom_port:
    import visdom
    visdom.Visdom(port=args.visdom_port).close()

from sacred.observers import MongoObserver


from sacred.observers import FileStorageObserver
name_ex="{}".format(args.model_name)
ex = Experiment(name_ex)
args.host=socket.gethostname()

if args.observe:
    dir_scripts="/remwork/filecremers2/dendorfp/trajnet/scripts/{}".format(args.host)
    if not os.path.exists(dir_scripts):
                os.makedirs(dir_scripts)
    ex.observers.append(MongoObserver.create( db_name='trajnet_{}'.format(args.host)))
    ex.observers.append(FileStorageObserver.create(dir_scripts))

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'



@ex.config
def configuration():
    args=args
@ex.capture
def get_info(_run):
    return  _run.experiment_info["name"], _run._id

def write_config(_run, dic):
    return  ex.add_config(dic)


@ex.main
def run_main(args):
    vis_env=get_info()
    args.environment=os.path.join(args.host, "{}_{}".format(vis_env[1],  vis_env[0])) 
   
    ex.info["vis_env"]=args.environment
    average, final, mean=main_func(args)
    ex.info["ADE"]= mean
    ex.info["FDE"]=final
    ex.info["AVERAGE"]=average

    return average
    
if __name__ == '__main__':
    run=ex.run()
  