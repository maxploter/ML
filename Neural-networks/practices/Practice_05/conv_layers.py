from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    STRIDE = conv_param['stride']
    PAD = conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_PRIME = int(1 + (H + 2 * PAD - HH) / STRIDE)
    W_PRIME = int(1 + (W + 2 * PAD - WW) / STRIDE)

    out = np.zeros((N, F, H_PRIME, W_PRIME))
    x = np.pad(x, ((0,0), (0,0), (PAD,PAD), (PAD,PAD)), mode='constant', constant_values=0)

    for img_num, img in enumerate(x):
        for i in range(H_PRIME):
            for j in range(W_PRIME):
                for f_num, f in enumerate(w):
                    receptive_h1 = i*STRIDE
                    receptive_h2 = receptive_h1 + HH
                    receptive_w1 = j*STRIDE
                    receptive_w2 = receptive_w1 + WW

                    receptive_field = img[:, receptive_h1:receptive_h2, receptive_w1:receptive_w2]

                    # print(receptive_field.shape)
                    # print(f.shape)
                    out[img_num, f_num, i, j] = np.sum(receptive_field * f) + b[f_num]

    x = x[:, :, PAD:PAD + H, PAD:PAD + W] # remove paddings

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    STRIDE, PAD = conv_param['stride'], conv_param['pad']
    N, F, H_PRIME, W_PRIME = dout.shape
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    x_pad = np.pad(x, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)), mode='constant', constant_values=0)
    dx_pad = np.pad(dx, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)), mode='constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for i in range(H_PRIME):
                for j in range(W_PRIME):
                    h_start = i * STRIDE
                    h_end = h_start + HH
                    w_start = j * STRIDE
                    w_end = w_start + WW

                    dw[f] += x_pad[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]
                    db[f] += dout[n, f, i, j]

    dx = dx_pad[:, :, PAD:PAD+H, PAD:PAD+W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    # Find the corners of the current "slice" (pooling window)
                    h_start = h * stride
                    h_end = h_start + pool_height
                    w_start = w * stride
                    w_end = w_start + pool_width

                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, h, w] = np.max(window)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + pool_height
                    w_start = w * stride
                    w_end = w_start + pool_width

                    window = x[n, c, h_start:h_end, w_start:w_end]

                    max_value = np.max(window)

                    mask = (window == max_value)

                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, h, w]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

