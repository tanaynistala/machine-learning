#
# behaviortree.py
#
# Defines the elements of a behavior tree for modular extension in Assignment 1
#
# Tanay Nistala
# 2022.09.20
#

# Result definitions
SUCCESS = "Success"
RUNNING = "Running"
FAILURE = "Failure"

# Node class structure
class Node:
	def __init__(self, func):
		self.func = func

	def run(self):
		return self.func()

# Node definitions

class Task(Node):
	pass

class Condition(Node):
	pass

class Composite(Node):
	def __init__(self, children):
		self.children = children

class Decorator(Node):
	def __init__(self, child):
		self.child = child

# Subnode definitions

## Composites

class Sequence(Composite):
	def run(self):
		for child in self.children:
			status = child.run()
			if status != SUCCESS:
				return status
		return SUCCESS

class Selection(Composite):
	def run(self):
		for child in self.children:
			status = child.run()
			if status != FAILURE:
				return status
		return FAILURE

class Priority(Composite):
	def run(self):
		self.children.sort(key=lambda x: x[1])
		for child in self.children:
			status = child[0].run()
			if status != FAILURE:
				return status
		return FAILURE

## Decorators

class Timer(Decorator):
	def __init__(self, time, child):
		super(Timer, self).__init__(child)
		self.time = time

	def run(self):
		if self.child.run() == FAILURE:
			return FAILURE
		self.time -= 1
		return RUNNING if self.time > 0 else SUCCESS

class RepeatUntilSuccess(Decorator):
	def run(self):
		if self.child.run() == SUCCESS:
			return SUCCESS
		return RUNNING

class RepeatUntilFailure(Decorator):
	def run(self):
		if self.child.run() == FAILURE:
			return SUCCESS
		return RUNNING

## Debug Node

class Debug(Node):
	def __init__(self, str):
		self.str = str

	def run(self):
		print(self.str)
		return SUCCESS
