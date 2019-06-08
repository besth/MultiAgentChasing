import pandas as pd
import pylab as plt
import numpy as np
from anytree import AnyNode as Node
from matplotlib import pyplot as plt
from datetime import datetime

# Local import
from algorithms.mcts import CalculateScore, SelectChild, Expand, RollOut, backup, GetActionPrior, MCTS, InitializeChildren, SelectNextAction
from envMujoco import Reset, TransitionFunction, IsTerminal
import reward
from runPolicyInMujoco import Evaluate


def computeDistance(pos1, pos2):
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, optimalNextPosition, nextPositionIndex, stateIndexInTuple, xPosIndex, numXPosEachAgent, agentID):
        self.optimalNextPosition = optimalNextPosition
        self.nextPositionIndex = nextPositionIndex
        self.stateIndexInTuple = stateIndexInTuple
        self.xPosIndex = xPosIndex
        self.numXPosEachAgent = numXPosEachAgent
        self.agentID = agentID

    def __call__(self, trajectories):
        statesAtNextStep = [trajectory[self.nextPositionIndex][self.stateIndexInTuple] for trajectory in trajectories]
        xPosAtNextStep = [state[self.agentID][self.xPosIndex:self.xPosIndex+self.numXPosEachAgent] for state in statesAtNextStep]
        distanceForAllTrajectories = [computeDistance(nextPosition, self.optimalNextPosition) for nextPosition in xPosAtNextStep]
        meanDistance = np.mean(distanceForAllTrajectories)
        distanceStdDev = np.std(distanceForAllTrajectories)

        return [meanDistance, distanceStdDev]


def drawPerformanceline(dataDf, axForDraw, xLabel, yLabel, yError):
    plotDf = dataDf.reset_index()

    for key, grp in plotDf.groupby(['rolloutHeuristicWeight']):
        if key == 0:
            label = 'Without Heuristic'
        else:
            label = 'With Heuristic'
        grp.plot(ax=axForDraw, kind='line', x=xLabel, y=yLabel, label=label, yerr=yError)


def main():
    startTime = datetime.now()

    sheepXPosInit = [0]
    sheepYPosInit = [0]
    wolfXPosInit = [8]
    wolfYPosInit = [0]
    cInit = [1]
    cBase = [100]
    numSimulations = [500]#[50, 250, 500, 1000, 1500, 2000]
    numTrials = [200]
    maxRunningSteps = [25]
    sheepPolicyName = ['mcts']
    render = [False]
    killzoneRadius = [0.5]
    aliveBonus = [-0.05]
    deathPenalty = [1]
    rolloutHeuristicWeight = [0.1, 0]

    levelValues = [sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit, cInit, cBase, numSimulations, numTrials, maxRunningSteps, sheepPolicyName, render, killzoneRadius, aliveBonus, deathPenalty, rolloutHeuristicWeight]
    levelNames = ['sheepXPosInit', 'sheepYPosInit', 'wolfXPosInit', 'wolfYPosInit', 'cInit', 'cBase', 'numSimulations',
                      'numTrials', 'maxRunningSteps', 'sheepPolicyName', 'render', 'killzoneRadius', 'aliveBonus', 'deathPenalty',
                      'rolloutHeuristicWeight']

    measurementMetricName1 = 'meanDistance'
    measurementMetricName2 = 'distanceStdDev'
    measurementMetricNames = [measurementMetricName1, measurementMetricName2]
    variableParameter1 = 'numSimulations'
    variableParameter2 = 'rolloutHeuristicWeight'

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

    evaluate = Evaluate(measurementFunction, measurementMetricNames, qVelInit, qPosInitNoise, qVelInitNoise)

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    print(toSplitFrame)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluate)
    print(modelResultDf)

    fig = plt.figure()

    plotRowNum = 1
    plotColNum = 1
    plotCounter = 1

    levelNames.remove(variableParameter1)
    levelNames.remove(variableParameter2)

    for (key, dataDf) in modelResultDf.groupby(levelNames):
        axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
        drawPerformanceline(dataDf, axForDraw, variableParameter1, measurementMetricName1, measurementMetricName2)
        plotCounter += 1

    plt.legend(loc = 'best')
    plt.show()

    endTime = datetime.now()
    print("Time taken: ", endTime-startTime)


if __name__ == "__main__":
    main()
