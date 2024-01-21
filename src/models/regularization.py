import torch

class Regularization_L1_NOTWORKING:
    def __init__(self, reg_strength : float = 0.001) -> None:
        self.reg_strength = reg_strength
    
    def __call__(self, model_parameters: torch.Generator) -> float:

        # let's use torch.linalg.norm instead of torch.sum(torch.abs(param))
        # It is faster
        lasso_reg = torch.linalg.norm(torch.nn.utils.parameters_to_vector(model_parameters), 1)
        return lasso_reg * self.reg_strength


class Regularization_L2_NOTWORKING:
    def __init__(self, reg_strength : float = 0.001) -> None:
        self.reg_strength = reg_strength
    
    def __call__(self, model_parameters: torch.Generator) -> float:
        l2_reg = torch.sum(torch.nn.utils.parameters_to_vector(model_parameters).pow(2)) 
        return l2_reg * self.reg_strength

class Regularization_L1:
    def __init__(self, reg_strength : float = 0.001) -> None:
        self.reg_strength = reg_strength
    
    def __call__(self, model_parameters: torch.Generator) -> float:

        # let's use torch.linalg.norm instead of torch.sum(torch.abs(param))
        # It is faster
        lasso_reg = sum(torch.linalg.norm(p, 1) for p in model_parameters)
        return lasso_reg * self.reg_strength

class Regularization_L2:
    def __init__(self, reg_strenght : float = 0.001) -> None:
        self.reg_strenght = reg_strenght
    
    def __call__(self, model_paramaters: torch.Tensor) -> float:
        l2_reg = torch.sum(torch.sum(param.pow(2)) for param in model_paramaters)
        return l2_reg * self.reg_strength
