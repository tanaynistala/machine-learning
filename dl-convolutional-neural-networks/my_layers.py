import numpy as np

# NOTE: it is easier for you to implement the convolution operation and the pooling operation with
#       for-loops. Backpropagation and running speed are not in our consideration in this task.
#

def conv_forward(input, filters, bias, stride, padding):
    """
    The purpose of the implementation is to match the Torch convolution operation, so you will know the low-level
    computation behind it. Please only use multiplications and additions from the numpy package.
    Please consult the documentation of `torch.nn.functional.conv2d`
    [link](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    for the calculation and arguments for this operation.

    We are considering a simpler case: the `input` is always in the format "NCHW".
    We only consider two padding cases: "SAME" and "VALID". We did not discuss groups and dilation,
    which both take their default value 1.

    """
    # TODO: Please implement the forward pass of the convolutional operation.
    # NOTE: you will need to compute a few sizes -- a handdrawing is useful for you to do the calculation.

    N, C_in, H, W = input.shape
    H_conv, W_conv = filters.shape[2:]

    if padding.upper() == "VALID":
        H_out = (H - H_conv) // stride[0] + 1
        W_out = (W - W_conv) // stride[1] + 1
    elif padding.upper() == "SAME":
        H_out = H // stride[0]
        W_out = W // stride[1]

        input = np.pad(input, ((0, 0), (0, 0), (H_conv // 2, H_conv // 2), (W_conv // 2, W_conv // 2)), 'constant')

    C_out = filters.shape[0]
    out = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, c, i, j] = np.sum(input[n, :, i:i+H_conv, j:j+W_conv] * filters[c, :, :, :]) + bias[c]

    return out



def max_pool_forward(input, ksize, stride, padding = "VALID"): # No need for padding argument here
    """
    The purpose of the implementation is to match the Torch pooling operation.
    Please consult the documentation of `torch.nn.MaxPool2d`
    [link](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    for the calculation and arguments for this operation.

    We are considering a simpler case: the `input` is always in the format "NCHW".
    We only consider two padding cases: "SAME" and "VALID".


    """

    # TODO: Please implement the forward pass of the max-pooling operation
    N, C, H, W = input.shape
    H_pool, W_pool = ksize

    if padding.upper() == "VALID":
        H_out = (H - H_pool) // stride[0] + 1
        W_out = (W - W_pool) // stride[1] + 1
    elif padding.upper() == "SAME":
        H_out = H // stride[0]
        W_out = W // stride[1]

        input = np.pad(input, ((0, 0), (0, 0), (H_pool // 2, H_pool // 2), (W_pool // 2, W_pool // 2)), 'constant')

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, c, i, j] = np.max(input[n, c, i*stride[0]:i*stride[0]+H_pool, j*stride[1]:j*stride[1]+W_pool])

    return out
