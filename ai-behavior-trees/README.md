# Assignment 1

The program makes the following assumptions in its execution:

- For every one second of activity, the battery level drops by 1%.
- There is a 10% chance of finding a dusty spot on every iteration.
- There is a 5% chance of the cleaning operation failing to signify that the cleaning operation is complete.
- As the robot is required to dock when the battery drops below 30%, it typically runs a cleaning operation at 30% to bring the battery level below the threshold.
- Docking instantly brings the robot's battery level up to 100%.
- By default, the robot receives no commands, but the program reads commands from the beginning.
- Home path routing is ignored as there is no spacial component at the moment. The `homePath` variable exists, but is never used.

## Structure

The project is divided into three files with varying levels of abstraction:

- `behaviortree.py`: Implements the core behavior tree logic, designed for reusability. Contains the following structure underneath the `Node` class:

- Node
	- Task
	- Condition
	- Composite
		- Sequence
		- Selection
		- Priority
	- Decorator
		- Timer
		- RepeatUntilSuccess
		- RepeatUntilFailure
	- Debug (only for testing the tree)

- `agent.py`: Implements the reflex agent for this assignment, built off of the behavior tree components listed above. Main branches of the tree are assembled separately, and is completely assembled into one Priority node.

- `main.py`: Implements a user interface layer for interaction with the agent.

## Execution

To execute the program, run `python3 main.py` with all three files in the same directory.

There are three commands that can be run:

- `spot`: Run a spot cleaning operation for 20 seconds
- `clean`: Run a general cleaning operation until the floor is cleaned or the battery level drops below 30%.
- `dock`: Return the robot to the home base and dock.

In addition, providing no input simply triggers a new evaluation of the behavior tree, advancing one iteration further. Inputs are retrieved every five iterations.
