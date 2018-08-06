import numpy as np
import torch
import logging
import utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def eval(net,  ep, loader, args,  config, phase):
    if phase in ["validation", "test"]:
        pass
    else: 
        logger.error("phase = {}: wrong input, Choose validation or test".format(phase))
        return -1
  
    for batch in loader:
        net.eval()
       
       
        val, y = net.predict(batch)
        logger.debug('validate out {0}, target {1}'.format(val.size(), y.size()))
   
        loss_mean=utils.ADE(val, y)
        loss_final = utils.FDE(val, y)
        loss_average=utils.AVERAGE(val, y)

        print('EPOCH: {} - {}: Final. Loss: {:.5f}'.format(ep, phase, loss_final.item()))
        print('EPOCH: {} - {}: Mean. Loss: {:.5f}'.format(ep, phase, loss_mean.item()))
        print('EPOCH: {} - {}: AVERAGE.  Loss: {:.5f}'.format(ep, phase, loss_average.item()))

        """
        vis.line(
        X=torch.cat((val[:,0, 0 ].unsqueeze(1), y[:,0, 0 ].unsqueeze(1)), 1), 
        Y=torch.cat((val[:,0, 1 ].unsqueeze(1),y[:,0, 1 ].unsqueeze(1)),1), 
        name=phase,
        win=phase,
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
        """

        net.train()
        return loss_average.item(), loss_final.item(), loss_mean.item()
    