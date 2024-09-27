#
# main.py
#
# Implements a Naïve Bayesian Classification
#
# Tanay Nistala
# 2022.11.20
#

import csv
import argparse
import math
import statistics as stat

def prob(model, observation, givenState):
	return model[givenState][int(observation * 2)]

def getProbs(model, obs):
	timespan = len(obs)

	# Initialize probability lists
	rawProbs = [ [0.5, 0.5] for _ in range(timespan) ]
	probs = rawProbs

	if not math.isnan(obs[0]):
		# Calculate initial probabilities
		rawProbs[0] = [(
			model[state][int(obs[0]*2)] * 				# Probability of first observation
			0.5											# Starting classification
		) for state in [0, 1]]							# Checks probabilities for both states

		# Normalize probabilities
		probs[0] = [
			0.0 if rawProb == 0 						# Default case for division by zero
			  else rawProb / sum(rawProbs[0])   		# Normalized probability
		for rawProb in rawProbs[0]]

	# Iterate over timespan
	for time in range(1, timespan):
		if not math.isnan(obs[time]):
			# Calculate variance
			varProb = fluctuations[int(stat.variance(obs[:time]) < 1)] if time > 1 else [ 0.5, 0.5 ]

			# Calculate raw probabilities
			rawProbs[time] = [
				prob(model, obs[time], state) *			# Probability of observation given state
				varProb[state] * 						# Probability of state given variance
				sum([
					transitions[state][prevState] *		# Transition probability
					probs[time-1][prevState]			# Probability of previous state
				for prevState in [0, 1]])
			for state in [0, 1] ]

			# Normalize probabilities
			probs[time] = [
				0.0 if rawProb == 0 					# Default case for division by zero
				  else rawProb / sum(rawProbs[time])	# Normalized probability
			for rawProb in rawProbs[time]]

	return probs

# Retrieves the state sequence for the given probability
def getStateSequence(probSequence):
	return map(lambda prob: prob[1], probSequence)

# Transition Probabilities
transitions = [
	[ 0.9, 0.1 ],		# bird -> bird		plane -> bird
	[ 0.1, 0.9 ]		# bird -> plane		plane -> plane
]

# Fluctuation Probabilities
fluctuations = [
	[ 0.9, 0.1 ],		# high variance -> bird		high variance -> plane
	[ 0.1, 0.9 ]		# low variance -> bird		low variance -> plane
]

# Creates an argument parser for the file inputs
parser = argparse.ArgumentParser(description="A radar trace classifier")
parser.add_argument("-m",
	dest="model",
	help="CSV file with probability distributions",
)
parser.add_argument("-o",
	dest="observations",
	help="CSV file with observation tracks",
)

args = parser.parse_args()

# Read in the model
model = []
with open(args.model) as csvfile:
	reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
	for row in reader:
		model.append([float(x) for x in row])

# Read in the observation tracks
obsTracks = []
with open(args.observations) as csvfile:
	reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
	for row in reader:
		obsTracks.append([float(x) for x in row if not math.isnan(float(x))])

# Iterate over each observation track
for track in obsTracks:
	# Find the solution for the track
	result = [*getStateSequence(getProbs(model, track))]
	print("Plane" if result[-1] > 0.5 else "Bird")
