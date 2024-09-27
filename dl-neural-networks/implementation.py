"""
In this assignment, you will implement feedforward neural network (multilayer perceptron) to perform two tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader


# This class is implemented for you to hold datasets. You may choose to use it or not.

class MyDataset(Dataset):
    def __init__(self, x, y, pr_type):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = (torch.long if pr_type == "classification"
                                                     else torch.float32))
    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# NOTE: you need to complete two classes and one function below. Please see instructions marked by "TODO".

class DenseLayer(nn.Module):
    """
    Implement a dense layer
    """
    def __init__(self, input_dim, output_dim, activation):

        """
        Initialize weights of the DenseLayer.
        args:
            input_dim: integer, the dimension of the input layer
            output_dim: integer, the dimension of the output layer
            activation: string, can be 'linear', 'relu', 'tanh', 'sigmoid', or 'softmax'.
                        It specifies the activation function of the layer
            param_init: this input is used by autograder. Please do NOT touch it.
        """

        super(DenseLayer, self).__init__()

        # TODO: The weight matrix W should have shape (input_dim, output_dim)
        # The calculation later should be X * W + b. Here * represents matrix multiplication
        # If X has shape (batch_size, input_dim), then the linear transformation should give shape (batch_size, output_dim)

        # TODO: Please do your initialization here. Bad initializations may lead to bad performance later.
        # Please do not change the two variable names because they will be used by the autograder

        self.W = nn.Parameter(torch.rand((input_dim, output_dim), dtype = torch.float32))
        self.b = nn.Parameter(torch.zeros(output_dim, dtype = torch.float32))

        # TODO: record input arguments and initialize other necessary variables
        self.input_dim  = input_dim
        self.output_dim = output_dim

        if activation == 'linear':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        This function implement the `forward` function of the dense layer
        """

        # TODO: implement the linear transformation
        outputs = inputs @ self.W + self.b

        # TODO: implement the activation function,
        outputs = self.activation(outputs)

        return outputs


class Feedforward(nn.Module):

    """
    A feedforward neural network.
    """

    def __init__(self, task_type, input_size, depth, hidden_sizes, output_size, reg_weight):

        """
        Initialize the model. This way of specifying the model architecture is clumsy, but let's use this straightforward
        programming interface so it is easier to see the structure of the program. Later when you program with torch
        layers, you will see more precise approaches of specifying network architectures.

        args:
          task_type: string, 'regression' or 'classification'. The task type.
          input_size: integer, the dimension of the input.
          depth:  integer, the depth of the neural network, or the number of connection layers.
          hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                        The first number is the number of output units of first connection layer, and so on so forth.
          output_size: integer, the number of classes. In our regression problem, please use 1.
          reg_weight: float, the regularization strength.

        """

        super(Feedforward, self).__init__()

        # Add a condition to make the program robust
        if not (depth - len(hidden_sizes)) == 1:
            raise Exception("The depth (%d) of the network should be 1 larger than `hidden_sizes` (%d)." % (depth, len(hidden_sizes)))

        # TODO: install all connection layers except the last one

        # Initialize the layer list
        self.layers = nn.ModuleList()

        # Add the first layer
        self.layers.append(DenseLayer(input_size, hidden_sizes[0], 'tanh'))

        # Add the hidden layers
        for i in range(depth - 2):
            self.layers.append(DenseLayer(hidden_sizes[i], hidden_sizes[i+1], 'linear'))
            self.layers.append(DenseLayer(hidden_sizes[i+1], hidden_sizes[i+1], 'tanh'))

        # Add the last layer
        self.layers.append(DenseLayer(hidden_sizes[-1], output_size, 'linear'))
        if task_type == 'classification':
            self.layers.append(DenseLayer(output_size, output_size, 'softmax'))

        self.reg_weight = reg_weight


    def forward(self, inputs):
        """
        Implement the forward function of the network.
        """

        #TODO: apply the network function to the input
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

    def calculate_reg_term(self):
        """
        Compute a regularization term from all model parameters

        args:
        """

        #
        # TODO: computer the regularization term from all connection weights
        # Note: there is a convenient alternatives for L2 norm: using weight decay. Here we consider a general approach, which
        # can calculate different types of regularization terms.
        # Please apply regularization weight here. This term should be directly added to the loss term

        reg_term = 0
        for layer in self.layers:
            reg_term += self.reg_weight * torch.norm(layer.W, p=2)

        return reg_term

def train(x_train, y_train, x_val, y_val, model, num_train_epochs, batch_size, lr, task_type):

    """
    Train this neural network using stochastic gradient descent.

    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets
      model: a torch module
      num_train_epochs: int, the number of training epochs.
      batch_size: int, the batch size
      lr: float, learning rate
      task_type: string, 'regression' or 'classification', the type of the learning task.
    """


    # TODO: set up dataloaders for the training set and the test set. Please check the DataLoader class. You will need to specify
    # your batch sizes there.

    train_dataset = MyDataset(x_train, y_train, task_type)
    val_dataset = MyDataset(x_val, y_val, task_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # TODO: initialize an optimizer. You can check the documentation of torch.optim.SGD, but you can certainly try more advanced
    # optimizers

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # TODO: decide the loss for the learning problem. Note that the training objective is the classification/regression loss
    # plus the regularization term (which has been weighted in your model)

    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif task_type == 'regression':
        criterion = nn.MSELoss()

    # TODO: train the model and record the training history after every training epoch

    history = {"loss": [],     # each entry of the list should be the average training loss over the epoch
              "val_loss": [],  # each entry of the list should be the validation loss over the validation set after the epoch
              "accuracy": []}  # each entry of the list should be the evaluation of the model (e.g. accuracy or MSE) over the
                               # validation set after the epoch.

    for epoch in range(num_train_epochs):
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss += model.calculate_reg_term()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["loss"].append(train_loss)

        model.eval()
        val_loss = 0
        val_preds = []

        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_preds.append(outputs)

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        val_preds = torch.cat(val_preds, dim=0)
        if task_type == 'classification':
            val_preds = torch.argmax(val_preds, dim=1)
            val_accuracy = accuracy_score(y_val, val_preds)
            history["accuracy"].append(val_accuracy)
        elif task_type == 'regression':
            val_preds = val_preds.detach().numpy()
            val_mse = np.mean((val_preds - y_val)**2)
            history["accuracy"].append(val_mse)

        print("Epoch %d" % epoch, end="\r")

    print("Training finished.")
    print("Final training loss: %.4f" % history["loss"][-1])
    print("Final validation loss: %.4f" % history["val_loss"][-1])
    if task_type == 'classification':
        print("Final validation accuracy: %.4f" % history["accuracy"][-1])
    elif task_type == 'regression':
        print("Final validation MSE: %.4f" % history["accuracy"][-1])

    return model, history
