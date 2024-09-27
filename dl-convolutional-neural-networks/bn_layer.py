import torch
import torch.nn as nn

class BNLayer(nn.Module):
    """
    In this problem, you are supposed to implement the batch normalization layer to match torch calculation.
    You are supposed to use elementry operations such as tensor additions and multiplications. You cannot
    import any Torch layers to do the calculation here.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Please consult the documentation of batch normalization
        (https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
        for the meaning of the arguments.

        Here you are asked to implementing a simpler version of the layer. Here are differences.
        1. This implementation will be applied to a CNN's feature map with format N,C,H,W. so a channel of a
           feature shares a set of parameters.

        2. The initializers both arrays, so you can use them directly. (tensorflow has special initializer
           classes)
        """

        super().__init__()

        # TODO: please complete the initialization of the class and take all these options
        self.running_mean = torch.zeros((1, num_features, 1, 1))
        self.running_var  = torch.ones((1, num_features, 1, 1))

        self.gamma    = torch.ones((1, num_features, 1, 1))
        self.beta     = torch.zeros((1, num_features, 1, 1))
        self.momentum = momentum
        self.eps      = eps


    def forward(self, inputs):
        """
        Implement the forward calculation during training and inference (testing) stages.
        """

        # TODO: Please implement the caluclation of batch normalization in training and testing stages

        # NOTE 1. you can use the binary flag `self.training` to check whether the layer works in the training or testing stage
        # If training, then: 1) calculate the mean and variance from the batch; 2) calculate the output according to the equation
        # and 3) update the running statistics (the mean and variance) using momentum  x <- (1 − momentum) × (previous x) + momentum × (current x)
        # If testing, the calculate the output according to the equation.

        if self.training:
            # Calculate the mean and variance from the batch
            mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
            var  = inputs.var(dim=(0, 2, 3), keepdim=True)

            # Calculate the output according to the equation
            x_hat = (inputs - mean) / torch.sqrt(var + self.eps)

            # Update the running statistics (the mean and variance) using momentum
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var

        else:
            # Calculate the output according to the equation
            x_hat = (inputs - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        outputs = self.gamma * x_hat + self.beta
        return outputs
