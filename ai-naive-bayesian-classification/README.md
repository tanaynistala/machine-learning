# Assignment 4

## Assumptions

- The program assumes that the model file provided contains two rows with an equal number of datapoints, akin to having two probability curves over the same range
- The program assumes that the observation file contains speed values within the velocity range of the model provided
- The model contains speeds in 0.5 m/s increments
- The observation contains datapoints captured once every second
- Birds are represented as `0` in the code, planes as `1`

## Structure

The project contains a `main.py` file with the code for the project. It contains the following functions:

- `prob`: returns the probability of the observation given the state (bird/plane)
- `getProbs`: iteratively calculates the probabilities of an observation being a bird or a plane over a dataset
- `getStateSequence`: retrieves the estimated state at every datapoint

## Execution

To execute the program, run `python3 main.py -m [modelFile] -o [observationFile]`.

`modelFile` is a CSV file that contains two rows with the probability curve for birds and planes, in that order.
`observationFile` is a CSV file that contains an observation stream on each row with values within the range of the model.

The program evaluates the data streams in order and outputs whether it is likely a bird or a plane for each.
