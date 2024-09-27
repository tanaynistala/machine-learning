#
# main.py
#
# Implements a constraint satisfaction problem to solve a sudoku
#
# Tanay Nistala
# 2022.11.13
#

import numpy as np
import time
import csv
import argparse

# BOARD STRUCTURE

# 3x3 BLOCK STRUCTURE
#
# 	0 	1 	2
# 	3 	4 	5
# 	6 	7 	8
#

hardPuzzle = np.array([
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],

	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],

	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ],
	[  0, 0, 0, 	0, 0, 0, 	0, 0, 0  ]
])

# FUNCTION DEFINITIONS

# Retrieves the domains for every cell on the board
def getDomainSizes(board):
	# Initialize empty domain board with largest domains possible
	domainBoard = np.array([[ 9 for col in range(9) ] for row in range(9)])

	for row in range(9):
		for col in range(9):
			# Retrieves domain if the cell is not preset
			if board[row, col] == 0:
				domainBoard[row, col] = len(getDomain(board, row, col))

	return domainBoard

# Calculates the domain for the provided cell
def getDomain(board, row, col):
	rowContents = set(board[row, :])	# Contents of other cells in the row
	colContents = set(board[:, col])	# Contents of other cells in the column
	blockContents = set(				# Contents of other cells in the block
		board[
			3*int(row/3) : 3*int(row/3)+3,
			3*int(col/3) : 3*int(col/3)+3
		].flatten()
	)

	# Calculates domain from numbers not in the row, column, and block
	domain = set(range(1, 10)) - rowContents - colContents - blockContents
	return domain

# Checks the validity of the row, column, and block of the provided cell
def checkBoard(board, row, col):
	if (
		checkSegment(board, row=row)						# Checks validity of the row
	and checkSegment(board, col=col)						# Checks validity of the column
	and checkSegment(board, block=3*int(row/3)+int(col/3))	# Checks validity of the block
	):
		return True
	return False

# Checks the validity of a row, column, or block
def checkSegment(board, row=(-1), col=(-1), block=(-1)):
	if row >= 0:
		return checkValidity(board[row,:])

	elif col >= 0:
		return checkValidity(board[:,col])

	elif block >= 0:
		return checkValidity(board[
			3*int(block / 3) : 3*int(block / 3) + 3,
			3*int(block % 3) : 3*int(block % 3) + 3
		])

# Verifies that the provided set has no duplicates (i.e. the set is valid)
def checkValidity(nums):
	# Retrieves the number of occurrences of each number
	_, counts = np.unique(nums, return_counts=True)
	return max(counts[1:]) == 1

# Utility function for printing the board
def printBoard(board, deltaRow=(-1)):
	# Line templates
	lineTop  = "╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗"		# Top edge
	lineNum  = "║ # │ # │ # ║ # │ # │ # ║ # │ # │ # ║"		# Number row
	lineMin  = "╟───┼───┼───╫───┼───┼───╫───┼───┼───╢"		# Minor divider
	lineMaj  = "╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣"		# Major divider
	lineBot  = "╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝"		# Bottom edge

	# Cursor movement template
	cursorUp = lambda lines: '\x1b[{0}A'.format(lines)

	# Parses the board into strings
	symbol = " 1234567890"
	nums   = [ [""]+[symbol[cell] for cell in row] for row in board ]

	if deltaRow==(-1):
		# Prints the top border of the sudoku grid
		print(lineTop)
	else:
		# Moves cursor to the row to be reprinted
		print(cursorUp(2*(9-deltaRow) + 1))

	for row in range(1 if deltaRow==(-1) else deltaRow+1, 10):
		# Prints the row of numbers
		print( "".join(n+s for n,s in zip(nums[row-1], lineNum.split("#"))) )
		# Prints the following divider
		print([ lineMin, lineMaj, lineBot ][ (row%9 == 0) + (row%3 == 0) ])

# Finds the cell with the smallest domain to be targeted next in the search
def getTarget(board):
	# Gets domains for every cell
	domainSizes = getDomainSizes(board)

	# Finds the cell location(s) with the smallest domain
	minRow, minCol = np.where(domainSizes == np.min(domainSizes))

	return minRow[0], minCol[0]

# Solves the provided sudoku board (recursively)
def solve(board, row=(-1), col=(-1)):
	# Sets a target if none provided (base case)
	if row == -1 and col == -1:
		row, col = getTarget(board)

	# Checks if the board is filled
	if np.min(board) > 0:
		return board

	# Checks that no domain is empty
	if np.min(getDomainSizes(board)) == 0:
		return

	# Loops over every number in the cell's domain
	for num in getDomain(board, row, col):
		board[row, col] = num				# Sets the cell's value to the number
		time.sleep(0.02)					# Uncomment this line to see the solver work a bit slower
		printBoard(board, deltaRow=row)		# Prints the board's new state

		# Checks that the new board is valid
		if checkBoard(board, row, col):
			# Gets the next target cell
			nextTargetRow, nextTargetCol = getTarget(board)

			# Recursively solves the new board with the new target cell
			result = solve(board, nextTargetRow, nextTargetCol)

			# Returns the board if the solution is found
			if result is not None:
				return result

		# Reverts changes if solution not found
		board[row, col] = 0

# MAIN RUNTIME

# Creates an argument parser for the file input
parser = argparse.ArgumentParser(description="A Sudoku puzzle solver")
parser.add_argument("-f",
	dest="file",
	help="CSV file with puzzle state",
)

args = parser.parse_args()

if args.file:
	results = []
	with open(args.file) as csvfile:
		reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
		for row in reader:
			results.append([int(x) for x in row])

	board = np.array(results)

	printBoard(board)
	result = solve(board)

	if result is None:
		print("No Solution Found.")
	else:
		print("Solution Found!")
else:
	print("ERROR: No file provided.")
