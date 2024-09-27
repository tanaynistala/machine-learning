# Assignment 6

## Assumptions

- The program assumes that the training data and query files provided contain five and four columns respectively, with the following:

  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
  - (Only for training data) Flower type

- Existence of `numpy` on the system

Unfortunately, I ran into several issues with the back-propagation algorithm which mean that the neural network is unable to correctly classify Versicolor flowers. I suspect there may be an issue with my math for the matrix multiplication that I used to carry out propagation, or that there may be an issue with how biases are calculated. However, the program does normally classify the other two types of flowers correctly.

There is also a number overflow issue that I was unable to solve, but it only results in warnings, not errors, so it is not a problem. It arises from cases where the sigmoid function evaluates close to zero.

## Structure

The project contains a `main.py` file with the code for the project. The Neural network is organized as a class with forward and backward propagation functions and helper sigmoid and sigmoid-derivative functions for activation. In addition there is a training function and a query function to conduct the class's function.

## Execution

To execute the program, run `python3 main.py -t [trainingFile] -q [queryFile]`.

`trainingFile` is a CSV file that contains five columns with the parameters listed above for a training set
`queryFile` is a CSV file that contains four columns with the parameters listed above to be queries

The program evaluates the query data and outputs the estimated type of flower
