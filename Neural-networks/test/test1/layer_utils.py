pass
from layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_arctan_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by an Arctan activation.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the Arctan activation
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, arctan_cache = arctan_forward(a)
    cache = (fc_cache, arctan_cache)
    return out, cache


def affine_arctan_backward(dout, cache):
    """
    Backward pass for the affine-arctan convenience layer.
    """
    fc_cache, arctan_cache = cache
    da = arctan_backward(dout, arctan_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db