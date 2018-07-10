import os
import itertools
import random
import multiprocessing as mp 
import numpy as np

epochs=200
models=['LSTM', 'GRU', "RNN"]
timewindow_val=[50, 100, 150, 200]
hidden_states_val=[25, 50, 75, 100]
layers_val=[1,2]

input=list(itertools.product(*[models, timewindow_val, hidden_states_val, layers_val]))
random.shuffle(input)

def job(nr, list_commands):
    for i in list_commands:
        os.system("CUDA_VISIBLE_DEVICE={} python run.py --train --e 25 --obs --m {} --lb {} --hs {} --nl {} --lf 1 --vp 8895".format(nr, i[0], i[1], i[2] , i[3]))

processes=[mp.Process(target=job, args=(x,input[x : (x+1)*4]))  for x in np.arange(4)]

# Run processes
for p in processes:
    p.start()

for p in processes:
    p.join()