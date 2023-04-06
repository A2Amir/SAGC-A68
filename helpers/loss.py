

import torch


class WeightedCrossEntropy_loss(torch.nn.Module):
    

    def __init__(self, class_weights, reduction='mean', ignore_index=255):
        super(WeightedCrossEntropy_loss, self).__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index = ignore_index)
        self.loss.weight = torch.tensor(class_weights, dtype=torch.float)#.to(self.device);

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c)
                target:(n, c)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        
        return self.loss(predict, target)
    
class Focal_loss(torch.nn.Module):
    


    def __init__(self, class_weights):
        super(Focal_loss, self).__init__()


        self.weight = class_weights
        
        self.loss = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='focal_loss',
                                    alpha= self.weight, gamma=2, reduction='mean', device='cpu',
                                    dtype=torch.float32, force_reload=False)

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c)
                target:(n, c)
                weight (list, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        
        return self.loss(predict, target)
    
     