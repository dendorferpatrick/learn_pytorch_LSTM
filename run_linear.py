import os
import numpy as np
import multiprocessing

def call_command(off):
    command="python run.py --seq 8 --pred 12 --e 0 --off {} --obs --model NN_linear --m linear_off{}".format( off, off)
    os.system(command)



off=np.arange(7)
for i in off:
    print(i)
    jobs=[]
    p = multiprocessing.Process(target=call_command, args=(i,))
    jobs.append(p)
    p.start()

