# Assignment 2

The problem is the found the optimal set of flips that puts the pancake stack in ascending order of size. The cost function for this search was defined to be the number of flips used to reach a state, while the heuristic function was defined to be the number of pancakes that are on top of one with an adjacent size.

The program in `main.py` implements an A\* algorithm to find a solution to this problem. A Uniform Cost Search (UCS) algorithm would also be a possible search method for this problem; however, it would have to expand by 10 states for every frontier node expanded, which would quickly exhaust most system resources, as every step multiplies the frontier size by 10.

## Assumptions

The program makes the following assumptions in its execution:

- The user will (mostly) enter correct inputs
- Flipping zero (0) or one (1) pancakes has no effect
- The cost of flipping any number of pancakes is the same i.e. there is no difference in cost between flipping 2 or 10 pancakes

## Structure

The project contains a `main.py` file with the code for the project. It contains the following functions:

- `printState`: prints a visual representation of the provided pancake stack.

- `flip`: executes a "flip" on the provided stack at the provided index.
- `expandFrontier`: explores all flips from the topmost stack on the frontier
- `checkGoal`: checks if the provided stack meets the goal criteria

- `heuristic`: calculates the distance from the provided stack to the goal
- `cost`: calculates the cost to reach the provided stack

The pancake stacks are set up as 11-element lists — the first element is a tuple containing the list of all flips taken to reach the current state, and the remaining 10 elements store the order of the stack. This was done to coordinate pancake numbers with their index numbers, and allows the flip history to be stored with the stack's state.

## Execution

The program requires the availability of the `heapdict` library, which implements a heap with decrease-key functionality. It can be installed using `pip3 install heapdict`. To execute the program, run `python3 main.py`.

1. The program asks for the size of the pancake stack. Defaults to 10, as specified in the assignment spec, but can be changed if required.
2. The program can set up the initial state of the pancake stack in two ways:
   - **Custom Stack**: (Enter `y` to use this mode) Allows the user to input the order for the initial stack.
   - **Randomized Stack**: (Enter `n` to use this mode) Allows the user to "shuffle" the stack for a random order. (Enter `n` when finished shuffling.)
3. Press `ENTER` to begin the A\* search algorithm. When done, the program outputs the path of flips found, and sets up a "replay" of the path. Press `ENTER` to walk through each step.
