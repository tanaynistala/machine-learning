#
# main.py
#
# Implements an Artificial Neural Network
#
# Tanay Nistala
# 2022.12.11
#

import csv
import argparse
import random
import numpy as np

class NeuralNetwork:
	# Initializes the neural network
	def __init__(self):
		self.inputSize  = 4		# INPUT
								#   ↓
		self.hiddenSize = 4		# HIDDEN
								#   ↓
		self.outputSize = 3		# OUTPUT

		self.layerCount = 1		# No. of hidden layers

		# Creates randomized weight and bias matrices
		inputWeights = np.random.randn(self.inputSize, self.hiddenSize)
		outputWeights = np.random.randn(self.hiddenSize, self.outputSize)

		self.weights = [
			inputWeights,
			*[np.random.randn(self.hiddenSize, self.hiddenSize) for layer in range(self.layerCount-1)],
			outputWeights
		]

		self.biases = [
			*[np.random.randn(self.hiddenSize, 1) for layer in range(self.layerCount)],
			np.random.randn(self.outputSize, 1)
		]

		# Initialize other model variables
		self.learningRate = 0.1
		self.error = 1.0

	# Trains the neural network
	def trainNetwork(self, inputLayer, outputLayerRef):
		# Loops for 1,000 iterations, or if the error is sufficiently low
		while self.learningRate > 0 and self.error > 0.001:
			# Loops over every input datapoint
			for layer in range(len(inputLayer)):
				# Generates the output using the current model
				outputLayer = self.frwdProp(inputLayer[layer].reshape((1,self.inputSize)))

				# Propagates backwards to handle error
				self.backProp(inputLayer[layer].reshape((1,self.inputSize)), outputLayer, outputLayerRef[layer])
				self.error = self.loss(outputLayer, outputLayerRef[layer])

			self.learningRate -= 0.001

	# Allows queries to run on the network
	def query(self, inputArr, outputArrRef=[]):
		output = self.frwdProp(inputArr)
		if len(outputArrRef) == 0:
			return output
		else:
			return output, self.loss(output, outputArrRef)

	# Forward propagation
	def frwdProp(self, inputLayer):
		self.layers = [inputLayer]

		# Computes and normalizes the hidden layers
		for layer in range(self.layerCount):
			self.layers.append(
				self.sigmoid(
					(self.layers[-1] @ self.weights[layer]) + self.biases[layer].T
				)
			)

		# Computes and normalizes the output layer
		outputLayer = self.sigmoid(self.layers[-1] @ self.weights[-1])

		return outputLayer

	# Backward propagation
	def backProp(self, inputLayer, outputLayer, outputLayerRef):
		# Computes the output layer error
		outputLayerErr = outputLayerRef - outputLayer
		outputLayerDel = outputLayerErr * self.dSigmoid(outputLayer)

		# Initializes the error array
		self.errors = [ (outputLayerErr, outputLayerDel) ]

		# Computes the error and delta for every layer
		for layer in range(self.layerCount):
			layerErr = self.errors[-1][1] @ self.weights[-1-layer].T
			layerDel = layerErr * self.dSigmoid(self.layers[-1-layer])

			self.errors.append((layerErr, layerDel))

		self.errors.reverse()

		# Adjusts the weights and biases
		for layer in range(self.layerCount+1):
			adjustments = self.layers[layer].T @ self.errors[layer][1]
			self.weights[layer] += adjustments

			self.biases[layer] += self.learningRate * self.errors[layer][1].T

	# Loss function
	def loss(self, outputLayer, outputLayerRef):
		return np.mean(np.square(outputLayerRef - outputLayer))

	# Activation function and derivative
	def sigmoid(self, val):
		return 1.0 / (1.0 + np.exp(-val))

	def dSigmoid(self, val):
		return self.sigmoid(val) * (1 - self.sigmoid(val))

# Creates an argument parser for the file inputs
parser = argparse.ArgumentParser(description="An iris flower classifier")
parser.add_argument("-t",
	dest="trainingData",
	help="CSV file with training data",
)
parser.add_argument("-q",
	dest="queryData",
	help="CSV file with query data",
)

args = parser.parse_args()

# Read in the training data
trainingInput = []
trainingOutput = []
types = []
with open(args.trainingData) as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		if row:
			type = row.pop()
			trainingOutput.append(type)
			trainingInput.append([float(x) for x in row])
			trainingInput

			if not type in types:
				types.append(type)

trainingInput = np.array(trainingInput, dtype=float)

# Map flower types to channels
trainingOutput = np.array([
	[
		1 if output == type else 0 for type in types
	] for output in trainingOutput
], dtype=float)

neuralNet = NeuralNetwork()
neuralNet.trainNetwork(trainingInput, trainingOutput)

# Read in the validation data
queryInput = []
queryOutput = []
with open(args.queryData) as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		if row:
			queryInput.append([float(x) for x in row])

queryInput = np.array(queryInput, dtype=float)

# Map flower types to channels
queryOutput = np.array([
	[
		1 if output == type else 0 for type in types
	] for output in queryOutput
], dtype=float)

totalLoss = 0
for queryNum in range(len(queryInput)):
	result, loss = neuralNet.query(queryInput[queryNum], queryOutput[queryNum])
	flowerType = types[result.argmax()]
	print(flowerType)
