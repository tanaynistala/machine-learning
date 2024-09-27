# Assignment 3

## Assumptions

The program makes the following assumptions in its execution:

- The user will (mostly) enter correct/valid inputs

## Structure

The project contains a `main.py` file with the code for the project. It contains the following functions:

- `printState`: prints a visual representation of the provided backpack's contents.

- `evolve`: evolves the provided population by one generation.
- `reproduce`: produces a child from the provided parent individuals, and mutates it based on a small probability.
- `mutate`: randomizes the genes of the provided individual to simulate a gene mutation.
- `cull`: culls the population in half by retaining the top half of the population, based on the fitness function.
- `fitness`: returns a fitness value for the provided backpack's contents

The fitness function returns a positive value between 0 and 1 if the provided backpack's weight does not exceed the limit; this value is equal to the backpack's value normalized by the total value of all available items. If the backpack's weight exceeds the limit, then the function returns a value between -1 and 0, found using the following equation:

fitness = ( weightLimit / backpackWeight ) - 1

This ensures that if no individuals meet the weight criteria, only those closer to meeting it move on to produce the next generation.

## Execution

To execute the program, run `python3 main.py`.

1. The program asks for the maximum weight allowed. Set it to 250 to use the assignment spec, but feel free to enter any value.
2. The program then asks for the number of items available. Set it to 12 to use the assignment spec, but feel free to enter any value.
3. The program can set up the list of items in two ways:
   - **Custom Set**: (Enter `y` to use this mode) Allows the user to input the weight and value for each available item. Use this to match the assignment spec, or try any other problem.
   - **Randomized Set**: (Enter `n` to use this mode) Produces a randomized list of items to use for the algorithm.
4. The genetic algorithm runs automatically, and prints a list of the items in the optimal backpack, as well as the total weight and value of the backpack.
