# Assignment 4

## Sources

- Unicode boxes for puzzle visualization modified from https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
- ANSI sequence for reprinting puzzle visualization modified from https://stackoverflow.com/questions/59506103/python-use-carriage-return-to-re-write-multiple-lines

## Assumptions

- The program assumes the general rules of Sudoku puzzles: every row, column, and 3x3 block must only contain the numbers 1 through 9, with no duplicates
- The puzzle fed into the puzzle is solvable; i.e. the initial state does not break the rules of the puzzle, and it has a unique solution

## Structure

The project contains a `main.py` file with the code for the project. It contains the following functions:

- `getDomainSizes`: returns a table containing the size of each cell's domain
- `getDomain`: returns the domain of the cell at the provided coordinates

- `checkBoard`: checks the validity of the row, column, and 3x3 block of the most recently changed cell
- `checkSegment`: helper function to check the validity of a row, column, or 3x3 block
- `checkValidity`: helper function to check that there are no duplicates in a segment

- `printBoard`: prints the current state of the board
- `getTarget`: determines the next cell to be targeted by finding the one with the smallest domain
- `solve`: solves the provided puzzle recursively

The Sudoku boards are set up as 2-dimensional, 9x9 `numpy` arrays. Rows and columns are zero-indexed, and 3x3 blocks are set up with the following indexing:

```
	0	1	2
	3	4	5
	6	7	8
```

Variable ordering is utilized to pick the next cell to target for the algorithm, and filtering backtracks from a state when any cell's domain becomes empty (i.e. it has no possible solutions).

## Execution

To execute the program, run `python3 main.py -f [puzzleFile]`.

`puzzleFile` is a CSV (`.csv`) file that contains the Sudoku puzzle to be solved. Sample CSV files have been submitted as `easyPuzzle.csv` and `hardPuzzle.csv`, plus one extra `extrahardPuzzle.csv`.

The program automatically runs the algorithm at full speed. For a better visualization of the algorithm's process, uncomment line `154`, which adds a delay between steps to slow down the algorithm.
