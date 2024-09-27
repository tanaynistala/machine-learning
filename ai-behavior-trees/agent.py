#
# agent.py
#
# Creates a reflex agent's behavior tree
#
# Tanay Nistala
# 2022.09.20
#

from behaviortree import *
import time
import random

# Agent class definition
class Agent:

	# Instantiates blackboard with provided values
	# Creates subtrees with major goals
	def __init__(self, bl, sc, gc, ds, hp):
		self.blackboard = {
			"batteryLevel": bl,
			"spotClean": sc,
			"generalClean": gc,
			"dustySpot": ds,
			"homePath": hp,
			"timer": 0
		}

		# Node definitions

		self.spotClean = Sequence([
				Condition(self.checkSpot),
				Timer(20, Task(self.cleanSpot)),
				Task(self.finishSpotClean)
			])

		self.dustyClean = Sequence([
				Condition(self.checkDusty),
				Timer(35, Task(self.cleanSpot))
			])

		self.generalClean = Sequence([
				Condition(self.checkGeneral),
				Sequence([
					Priority([
						[self.dustyClean, 1],
						[RepeatUntilFailure(Task(self.cleanFloor)), 2]
					]),
					Task(self.finishGeneralClean)
				])
			])

		self.docking = Sequence([
				Condition(self.checkBattery),
				Task(self.dock)
			])

	# Main execution function, takes a command as input
	def run(self, cmd = ""):
		if cmd == "spot":
			self.blackboard["spotClean"] = True
		elif cmd == "clean":
			self.blackboard["generalClean"] = True
		elif cmd == "dock":
			self.dock()
			return SUCCESS

		Priority([
			[self.docking, 1],
			[Selection([self.spotClean, self.generalClean]), 2]
		]).run()

	# Task functions

	def cleanSpot(self):
		time.sleep(1)
		self.drainBattery()
		return SUCCESS

	def cleanFloor(self):
		time.sleep(1)
		self.drainBattery()

		# 5% chance of completing every iteration
		return FAILURE if random.randint(1, 100) < 5 else SUCCESS

	def finishSpotClean(self):
		self.blackboard["spotClean"] = False
		print("Finished spot cleaning!")
		return SUCCESS

	def finishGeneralClean(self):
		self.blackboard["generalClean"] = False
		print("Finished cleaning!")
		return SUCCESS

	def dock(self):
		print("Docked!")
		self.blackboard["batteryLevel"] = 100
		return SUCCESS

	def drainBattery(self):
		self.blackboard["batteryLevel"] -= 1
		print("Battery: " + str(self.blackboard["batteryLevel"]) + "%")

	# Condition functions

	def checkBattery(self):
		return SUCCESS if self.blackboard["batteryLevel"] < 30 else FAILURE

	def checkSpot(self):
		return SUCCESS if self.blackboard["spotClean"] else FAILURE

	def checkGeneral(self):
		return SUCCESS if self.blackboard["generalClean"] else FAILURE

	def checkDusty(self):
		if self.blackboard["dustySpot"]:
			return SUCCESS

		# 10% chance of finding a dusty spot
		if random.randint(1, 100) < 10:
			self.blackboard["dustySpot"] = True
			print("Found a dusty spot.")
			return SUCCESS
		return FAILURE
