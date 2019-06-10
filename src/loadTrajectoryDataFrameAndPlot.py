import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from testMCTSHeuristicEffectInMujoco import drawPerformanceline, DistanceBetweenActualAndOptimalNextPosition
from envMujoco import Reset, IsTerminal, TransitionFunction


class ConvertTrajectoriesDfToListOfTuples:
    def __init__(self, numTrajectories):
            self.numTrajectories = numTrajectories

    def __call__(self, fileToReadFrom):
        idx = pd.IndexSlice
        pickleIn = open(fileToReadFrom, 'rb')
        df = pickle.load(pickleIn)

        trajectories = []
        for trajectory in range(self.numTrajectories):
            print("trajectory:", trajectory)
            trajectoryValues = df.loc(axis=0)[idx[:, :, trajectory]].to_numpy()
            trajectoryLength = np.shape(trajectoryValues)[0]
            trajectory = [(trajectoryValues[step][0], trajectoryValues[step][1]) for step in range(trajectoryLength)]
            trajectories.append(trajectory)

        return trajectories


def main():
    sheepXPosInit = [-8]
    sheepYPosInit = [0]
    wolfXPosInit = [8]
    wolfYPosInit = [0]
    numSimulationsAllValues = list(range(12))
    useHeuristicAllValues = [False]
    prior = 'nonUniform'

    measurementMetricName1 = 'meanDistance'
    measurementMetricName2 = 'distanceStdDev'
    variableParameter1 = 'numSimulations'
    variableParameter2 = 'prior'

    envModelName = 'twoAgents'
    qPosInit = [sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit]
    qPosInit = [item for sublist in qPosInit for item in sublist]
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
    qPosInitNoise = 0
    qVelInitNoise = 0
    minXDis = 0.5
    renderOn = False
    numSimulationsFrames = 20
    allAgentsOptimalActions = [[10, 0], [0, 0]]
    nextPositionIndex = 1
    stateIndexInTuple = 0
    xPosIndex = 2
    numXPosEachAgent = 2
    agentID = 0

    reset = Reset(envModelName, qPosInit, qVelInit, numAgents, qPosInitNoise, qVelInitNoise)
    initState = reset()
    isTerminal = IsTerminal(minXDis)
    transitionFunction = TransitionFunction(envModelName, isTerminal, renderOn, numSimulationsFrames)
    optimalNextState = transitionFunction(initState, allAgentsOptimalActions)
    optimalNextPosition = optimalNextState[agentID][xPosIndex:xPosIndex+numXPosEachAgent]
    measurementFunction = DistanceBetweenActualAndOptimalNextPosition(optimalNextPosition, nextPositionIndex, stateIndexInTuple, xPosIndex, numXPosEachAgent, agentID)

    convertTrajectoriesDfToListOfTuples = ConvertTrajectoriesDfToListOfTuples(50)

    measurementsDfListForAllConditions = []

    for numSimulations in numSimulationsAllValues:
        fileToReadFrom = "trajectories/NumSim{}UseHeuristicFalsePriornonUniform.pickle".format(numSimulations)
        trajectories = convertTrajectoriesDfToListOfTuples(fileToReadFrom)
        measurements = measurementFunction(trajectories)
        measurementsDf = pd.DataFrame({variableParameter1: [numSimulations], variableParameter2: [prior], measurementMetricName1: [measurements[0]], measurementMetricName2: [measurements[1]]})
        measurementsDf = measurementsDf.set_index([variableParameter1, variableParameter2])
        measurementsDfListForAllConditions.append(measurementsDf)

    measurementsDfAllConditions = pd.concat(measurementsDfListForAllConditions)

    fig = plt.figure()
    variableParameter1 = 'numSimulations'

    axForDraw = fig.add_subplot(1, 1, 1)
    drawPerformanceline(measurementsDfAllConditions, axForDraw, variableParameter1, measurementMetricName1, measurementMetricName2)
    plt.xlim((0, 13))
    plt.show()





if __name__ == '__main__':
    main()




