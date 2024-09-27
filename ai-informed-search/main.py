#
# main.py
#
# Implements an A* algorithm for solving the Pancake Problem.
#
# Tanay Nistala
# 2022.10.09
#

from heapdict import heapdict
import random

##########

# Prints a visual representation of the provided stack
def printState(stack):
	# Iterates over all pancakes in the stack (the first element in the stack is the flip history)
	for pancake in stack[1:]:
		print(pancake, end="\t")
		for i in range(pancake):
			print("◼︎", end="")
		print()

# Defines the heuristic function for the search
def heuristic(stack):
	sum = 0

	# Compares each pancake to the one directly below
	for i in range(1, stackSize):
		# Adds 1 if the next pancake is an adjacent size
		sum += 1 if abs( stack[i] - stack[i+1] ) > 1 else 0

	return sum

# Defines the cost function for the search
def cost(stack):
	# Returns the size of the flip history
	return len(stack[0])

# Flips the top <index> pancakes on the stack
def flip(stack, index):
	# Converts the stack into a list
	stackList = [*stack,]

	# Flips the first <index> pancakes (skips the first element in the stack since it is the flip history)
	stackList[ 1 : index+1 ] = stackList[ index : 0 : -1 ]

	# Updates the flip history
	stackList[0] = stackList[0] + (index,)

	# Converts the stack back into a tuple for return
	return (*stackList,)

# Explores all flips on the frontmost stack in the frontier
def expandFrontier(frontier):
	# Pops the frontmost stack in the frontier
	currentStack = frontier.popitem()[0]

	# Generate stacks for every possible flip location
	for flipLocation in range(2, stackSize+1):
		# Creates the new stack flipped at the given index
		newStack = flip(currentStack, flipLocation)

		# Calculates the total cost of the new stack
		newTotalCost = cost(newStack) + heuristic(newStack)

		# Adds the new stack to the frontier
		frontier[newStack] = newTotalCost

# Checks whether the provided stack satisfies the goal condition
def checkGoal(stack):
	return stack[1:] == (*range(1, stackSize+1),)

# Runs one iteration of node expansion and checks if the goal is reached
def run():
	# Checks if the frontier is empty
	if frontier.peekitem() == None:
		return False

	# Checks if the frontmost stack meets the goal condition
	if checkGoal(frontier.peekitem()[0]):
		return True

	# Expands the frontier from the frontmost stack
	expandFrontier(frontier)

##########

# Ask for the size of the stack (Default: 10)
stackSize = int(input("How many pancakes are you sorting? (Defaults to 10): ")) or 10

# Asks if stack is shuffled or custom
if input("Would you like a custom stack? (y/n): ")[0] == 'y':
	pancakes = []

	# Asks for the size of each pancake
	for i in range(stackSize):
		num = None

		# Asks for size until a valid size is entered
		while not num in range(1, stackSize+1) or num in pancakes:
			num = int(input(f"Enter the pancake size (1-{stackSize}, must be unique): "))

			# Alerts if the size was invalid
			if not num in range(1, stackSize+1):
				print("Size out of range! Try again.")
			elif num in pancakes:
				print("That size already exists! Try again.")

		# Adds the pancake to the list
		pancakes.append(num)
		print("Current Stack:", *pancakes)
		print()

else:
	# Creates a stack with numbers 1 through <stackSize>
	pancakes = [*range(1, stackSize+1),]
	print("Current Stack:", *pancakes)

	# Asks user to shuffle the stack
	while input("Shuffle the stack? (y/n): ")[0] == 'y':
		random.shuffle(pancakes)
		print("Current Stack:", *pancakes)

# Creates the initial stack
initialStack = tuple( [()] + pancakes )

# Initializes the frontier with the initial stack
frontier = heapdict()
frontier[initialStack] = 0

# Prints the initial stack
print("Initial Stack:")
printState(initialStack)

_ = input("Press ENTER to search for a solution:")

# Continually runs the search until the algorithm returns a solution or fails
result = None
while result is None:
	result = run()

if result:
	# Handles case where result was found
	print("Solution found!\n")

	# Stores the solution
	solution = frontier.peekitem()[0][0]

	# Prints an overview of the solution
	print("Flips:", *solution, "\n")

	# Prints initial state of the stack
	print("Initial State:")
	printState(initialStack)
	print()

	# Prints each step of the solution
	currentStack = initialStack
	for flipLocation in solution:
		# Waits to continue to next step
		_ = input("Press ENTER to continue:")

		# Flips the stack
		currentStack = flip(currentStack, flipLocation)

		# Prints the new stack
		print("Flipping", flipLocation, "pancakes.")
		printState(currentStack)
		print()

else:
	# Handles case where result was not found
	# (NOT THEORETICALLY POSSIBLE)
	print("No solution found.")
