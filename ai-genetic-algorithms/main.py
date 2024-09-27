#
# main.py
#
# Implements a genetic algorithm to find the optimal selection of valuables while minimizing weight.
#
# Tanay Nistala
# 2022.10.28
#

import random
from itertools import *
from functools import *

##########

# Prints the state of the provided backpack
def printState(backpack):
	print("\nCONTENTS")

	totalWeight = 0
	totalValue = 0
	for (isIncluded, item) in zip(backpack, items):
		if isIncluded:
			print("Weight:", item[0], "\tValue:", item[1])
			totalWeight += item[0]
			totalValue += item[1]

	print("\nTOTAL")
	print("Weight:", totalWeight, "\tValue:", totalValue)

# Produces a child from two parent states by splicing the parents' phenotypes
def reproduce(parent1, parent2):
	global itemCount
	spliceLoc = random.randint(0, itemCount)	# Selects a location to splice the parents
	return parent1[:spliceLoc] + parent2[spliceLoc:]		# Splices the parents together

# Mutates the provided individual by randomizing the phenotype
def mutate(individual):
	return [bool(random.getrandbits(1)) for x in range(itemCount)]

# Determines the fitness of the individual based on value and weight consumption
def fitness(individual):
	global items

	# Produces a list of all items in the current backpack
	backpackItems = list(compress(items, individual))

	# Calculates the total weight and value of the backpack
	currentWeight = sum(map(lambda x: x[0], backpackItems))
	currentValue = sum(map(lambda x: x[1], backpackItems))

	# Returns a positive fitness value if the weight is below the limit, else negative
	return (currentValue / maxValue) if currentWeight <= maxWeight else (maxWeight / currentWeight) - 1

# Culls the population by half based on the provided fitness evaluator function
def cull(population, evaluator):
	# Creates a sorted list of (individual, fitness) pairs for easy evaluation when culling
	evaluatedPopulation = list(map(lambda individual: (individual, evaluator(individual)), population))
	evaluatedPopulation.sort(key = lambda x: x[1])

	# Culls the population based on fitness
	culledPopulation = evaluatedPopulation[int(populationSize/2):]

	# Removes fitness values from returned list
	return list(map(lambda x: x[0], culledPopulation))

# Produces the next generation of the provided population
def evolve(population, evaluator):
	nextGeneration = []

	# Culls population by 50%
	culledPopulation = cull(population, evaluator)

	for i in range(populationSize):
		# Picks two random individuals to reproduce
		parent1 = random.choice(culledPopulation)
		parent2 = random.choice(culledPopulation)

		# Reproduces the parents to produce a child
		child = reproduce(parent1, parent2)

		# Mutate the child on a random probability
		if random.random() < 0.05:
			mutate(child)

		# Adds the child to the next generation
		nextGeneration.append(child)

	nextGeneration.sort(key = lambda x: x[1])
	return nextGeneration

##########

populationSize = 10000
maxWeight = int(input("Maximum allowed weight: "))
maxValue = 0
itemCount = int(input("Number of items: "))
items = []

# Asks if items should be randomly generated or not
if input("Would you like a custom set of items? (y/n): ")[0] == 'y':
	for item in range(itemCount):
		# Gets details for the i'th item
		print("\nITEM #" + str(item+1))

		weight = int(input("Weight: "))		# Asks for weight
		value = int(input("Value: "))		# Asks for value
		maxValue += value

		items.append((weight, value))
else:
	print("#\tWeight\tValue")
	for item in range(itemCount):
		weight = random.randint(0, maxWeight)		# Randomizes weight
		value = random.randint(1, 100)		# Randomizes value
		maxValue += value

		items.append((weight, value))
		print(f"{item+1}\t{weight}\t{value}")

# Creates an initial population of randomized backpacks
initialPopulation = []
for individual in range(populationSize):
	backpack = [bool(random.getrandbits(1)) for x in range(itemCount)]
	initialPopulation.append(backpack)

population = initialPopulation
generation = 1

print("\nFinding a solution...")

# Loops the evolutionary process for at least 100 generations, and until a solution is found
while generation <= 100 or fitness(population[-1]) < 0:
	population = evolve(population, fitness)

	ending = '\r' if generation < 100 or fitness(population[-1]) < 0 else '\n'
	print(f'Currently at generation {generation}', end=ending)
	generation += 1

printState(population[-1])
