import numpy as np
import torch
import logging
import utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def validate(net, model_type,  ep, eval_loader, args, vis, look_back, config):
    print('-'* 20 + ' Validation ' +'-'*20)             
    
    for batch in eval_loader:
        net.eval()
       
       
        val, y = net.predict(batch)
        logger.debug('validate out {0}, target {1}'.format(val.size(), y.size()))
        #loss_final = net.criterion(val, y) 
        #loss_mean = net.criterion(val[-1], y[-1]) 
        loss_mean=utils.ADE(val, y)
        
        loss_final = utils.FDE(val, y)
        loss_average=utils.AVERAGE(val, y)
        print('{}: Final. Loss: {:.5f}'.format(model_type, loss_final.item()))
        print('{}: Mean. Loss: {:.5f}'.format(model_type, loss_mean.item()))
        print('{}: AVERAGE.  Loss: {:.5f}'.format(model_type, loss_average.item()))

        #ex.log_scalar("val_loss_{}".format(model_type), loss.item(), ep)
       
        vis.line(
        X=torch.cat((val[:,0, 0 ].unsqueeze(1), y[:,0, 0 ].unsqueeze(1)), 1), 
        Y=torch.cat((val[:,0, 1 ].unsqueeze(1),y[:,0, 1 ].unsqueeze(1)),1), 
        #update='replace',
        name="test",
        win="test",
        opts=dict(showlegend=True, 
            xtickmin=config.scale[1, 0].item(),
            xtickmax=config.scale[0, 0].item(),
            ytickmin=config.scale[1, 1].item(),
            ytickmax=config.scale[0, 1].item(),
            width=500, 
            height=500,
            legend=['Prediction', 'Ground truth'], 
            xlabel= 'x', 
            ylabel= 'y',) ,
                            )
        

        net.train()
        return loss_average.item(), loss_final.item(), loss_mean.item()
    