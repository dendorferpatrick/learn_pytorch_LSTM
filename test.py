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
import visdom


from sacred.observers import MongoObserver


from sacred.observers import FileStorageObserver

ex = Experiment('RNNs')
ex.observers.append(MongoObserver.create( db_name='GPUserver'))
ex.observers.append(FileStorageObserver.create('scripts'))

from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys'



@ex.config
def configuration():
    # Parameters
    features= 3

  
@ex.capture
def get_info(_run):
    return  _run.experiment_info["name"], _run._id

def write_config(_run, dic):
    return  ex.add_config(dic)


@ex.main
def run():
    print("hello world")
    ex.info["Result"]=1  

if __name__ == '__main__':
    run=ex.run()
    
    #run.root_logger = None
    #run.run_logger
