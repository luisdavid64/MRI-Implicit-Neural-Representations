import torch

class Regularization_L1_NOTWORKING:
    def __init__(self, reg_strength : float = 0.001) -> None:
        self.reg_strength = reg_strength
    
    def __call__(self, model_parameters: torch.Generator) -> float:

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

        lasso_reg = sum(torch.sum(torch.abs(p)) for p in model_parameters)
        return lasso_reg * self.reg_strength

class Regularization_L2:
    def __init__(self, reg_strength : float = 0.001) -> None:
        self.reg_strength = reg_strength
    
    def __call__(self, model_paramaters: torch.Generator) -> float:
        l2_reg = abs(sum(torch.sum(param.pow(2)) for param in model_paramaters))
        return l2_reg * self.reg_strength
