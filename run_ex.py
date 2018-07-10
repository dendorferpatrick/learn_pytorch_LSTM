import os
import itertools
import random
import multiprocessing as mp 
import numpy as np

epochs=200
mode=["m2o", "m2m"]
models=['LSTM', 'GRU', 'RNN']
timewindow_val=[50, 100, 150, 200]
hidden_states_val=[25, 50, 75, 100]
layers_val=[1,2]


input=list(itertools.product(*[mode, models, timewindow_val, hidden_states_val, layers_val]))
random.shuffle(input)

batch=len(input)/4
print(batch)
def job(nr, list_commands):
    for i in list_commands:
        os.system("CUDA_VISIBLE_DEVICE={} python run.py --train --e 50 --obs --mode {} --m {} --lb {} --hs {} --nl {} --lf 1 --vp 8895".format(nr, i[0], i[1], i[2] , i[3],  i[4]))

processes=[mp.Process(target=job, args=(x,input[int(x*batch) : int((x+1)*batch)]))  for x in np.arange(4)]

# Run processes
for p in processes:
    p.start()

for p in processes:
    p.join()