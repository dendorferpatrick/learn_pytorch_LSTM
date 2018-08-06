import glob
import numpy as np
import torch 
import os
from torch import nn
from NN import model
from torch.autograd import Variable

def get_txt_files(base_dir):
    return glob.iglob(f"{base_dir}/**/*.txt", recursive=True)


def test(model, epoch):
    path_data='/usr/wiss/dendorfp/dvl/projects/trajnet/data/test'
    
    path_output= '/usr/wiss/dendorfp/dvl/projects/trajnet/output/{0}_{1}'.format(model, epoch)
    if os.path.exists(path_output)==False:
        os.mkdir(path_output)
    FOLDERS=["stanford", "crowds", "biwi"]
    for folder in FOLDERS:
        path_folder=os.path.join(path_output, folder)
        if os.path.exists(path_folder)==False:
            os.mkdir(path_folder)
    
    os.chdir(path_data)

    # load model
    path_model=os.path.join("/usr/wiss/dendorfp/dvl/projects/trajnet/RNN/models", model, "epoch_{}.tar".format(epoch))
    model = torch.load(path_model)
    print("model loaded successfully")
    data = get_txt_files(".")
    print(data)
    model.eval()
    for txt_file in data:
        print(txt_file)
 
        test_data=np.genfromtxt(txt_file, delimiter=" ")
        pred=test_data.copy()
        labels=np.unique(test_data[:,1])
        seq_length= 8
        pred_length= 12
        index=0
        for id in labels:
            traj=test_data[test_data[:, 1]==id]
            x=Variable(torch.Tensor(traj[:seq_length, 2:])).float().cuda()
            x=x.unsqueeze(1)
            print(x.size())
            out=model.forward(x)
            print(out)
            pred[(index+seq_length):(index+seq_length+pred_length), 2:]= out
            index+=seq_length+ pred_length
        np.savetxt(os.path.join(path_output, txt_file), pred, delimiter=" ", fmt=('%i', '%.1f', '%.3f', '%.3f') )
        print("saved: %s" )


test("53_LSTM_hs50_nl2", 23)