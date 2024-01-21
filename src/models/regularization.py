import torch

class Regularization_L1:
    def __init__(self, reg_strenght : float = 0.001) -> None:
        self.reg_strenght = reg_strenght
    
    def __call__(self, model_paramaters: torch.Tensor):

        # let's use torch.linalg.norm instead of torch.sum(torch.abs(param))
        # It is faster
        lasso_reg = torch.sum(torch.linalg.norm(param, 1) for param in model_paramaters)
        return lasso_reg * self.reg_strength
    

class Regularization_L2:
    def __init__(self, reg_strenght : float = 0.001) -> None:
        self.reg_strenght = reg_strenght
    
    def __call__(self, model_paramaters: torch.Tensor) -> float:
        l2_reg = torch.sum(torch.sum(param.pow(2)) for param in model_paramaters)
        return l2_reg * self.reg_strength