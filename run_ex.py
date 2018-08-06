import multiprocessing
import itertools
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)


#parser.add_argument('--hs', dest='hidden_states',  default=False, help='Number of hidden states')  
parser.add_argument('--nn',dest='NN', type=int,  default = False, help='neural net model') 
parser.add_argument('--gpu',dest='GPU', type=int,  default= False, help='GPU') 
args = parser.parse_args()
print(args)


def call_command(input):
    if input[2]==2:
        model_name= "constant_velocity"
    elif input[2]==3:
        model_name= "variable_velocity"
    elif input[2]==1:
        model_name= "acceleration+const_velocity"
    command="CUDA_VISIBLE_DEVICES={} python run.py --seq 8 --pred 12 --e 500 --hs {}  --bs 8 --nl 1  --obs --model NN{} --m {}".format(input[0], input[1], input[2],  model_name)
    os.system(command)


if __name__ == '__main__':
    jobs = []
    GPU=[args.GPU]
    hidden_states=[5,15,25,40]
    NN=[args.NN]
    input=list(itertools.product(*[GPU, hidden_states, NN]))
    print(input)
    for i in input:
        print(i)
        p = multiprocessing.Process(target=call_command, args=(i,))
        jobs.append(p)
        p.start()

