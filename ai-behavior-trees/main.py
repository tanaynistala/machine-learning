#
# main.py
#
# Implements a user interface for the reflex agent
#
# Tanay Nistala
# 2022.09.20
#

from agent import *

# initial values
batteryLevel = 100
spotClean = False
generalClean = False
dustySpot = False
homePath = []

# create the agent
agent = Agent(
	batteryLevel,
	spotClean,
	generalClean,
	dustySpot,
	homePath
)

# command loop
cmd = input("Enter a command: ")
while cmd != "end":
	agent.run(cmd)
	for i in range(4):
		agent.run()
	cmd = input("Enter a command: ")
