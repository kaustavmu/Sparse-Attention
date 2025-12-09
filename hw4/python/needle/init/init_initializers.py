import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    # Xavier/Glorot uniform initialization
    # Uniform distribution in [-bound, bound] where bound = gain * sqrt(6 / (fan_in + fan_out))
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    # Check if shape is provided in kwargs
    if 'shape' in kwargs and kwargs['shape'] is not None:
        return rand(*kwargs['shape'], low=-bound, high=bound, **{k: v for k, v in kwargs.items() if k != 'shape'})
    else:
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    # Xavier/Glorot normal initialization
    # Normal distribution with std = gain * sqrt(2 / (fan_in + fan_out))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    # Check if shape is provided in kwargs
    if 'shape' in kwargs and kwargs['shape'] is not None:
        return randn(*kwargs['shape'], mean=0, std=std, **{k: v for k, v in kwargs.items() if k != 'shape'})
    else:
        return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Kaiming/He uniform initialization for ReLU
    # Uniform distribution in [-bound, bound] where bound = sqrt(6 / fan_in)
    bound = math.sqrt(6.0 / fan_in)
    
    # If shape is provided, use it instead of fan_in, fan_out
    if shape is not None:
        return rand(*shape, low=-bound, high=bound, **kwargs)
    else:
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Kaiming/He normal initialization for ReLU
    # Normal distribution with std = sqrt(2 / fan_in)
    std = math.sqrt(2.0 / fan_in)
    
    # Check if shape is provided in kwargs
    if 'shape' in kwargs and kwargs['shape'] is not None:
        return randn(*kwargs['shape'], mean=0, std=std, **{k: v for k, v in kwargs.items() if k != 'shape'})
    else:
        return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION