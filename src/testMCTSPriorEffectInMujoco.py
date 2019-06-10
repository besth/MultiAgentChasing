import numpy as np
import pickle
import pandas as pd
import pylab as plt
from matplotlib import pyplot as plt
from datetime import datetime

import skvideo
import skvideo.io

skvideo.setFFmpegPath("/usr/local/bin")

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, RolloutHeuristicBasedOnClosenessToTarget
import envMujoco as env
from envSheepChaseWolf import TransitionFunctionSheepSimulation, stationaryWolfPolicy, WolfPolicyForceDirectlyTowardsSheep, DistanceBetweenActualAndOptimalNextPosition
from loadTrajectoryDataFrameAndPlot import ConvertTrajectoriesDfToListOfTuples
from envMujoco import Reset, TransitionFunction, IsTerminal
import reward


def evaluateMeanEpisodeLength(trajectoriesAllTrials):
    episodeLengthAllTrials = [len(trajectory) for trajectory in trajectoriesAllTrials]
    meanEpisodeLength = np.mean(episodeLengthAllTrials)

    return meanEpisodeLength


class ConvertTrajectoryToDf:
    def __init__(self, numSimulations, useHeuristic):
        self.numSimulations = [numSimulations]
        self.useHeuristic = [useHeuristic]

    def __call__(self, trajectoryIndex, trajectory):
        trajectoryIndex = [trajectoryIndex]
        timeStep = range(len(trajectory))

        levelNames = ['numSimulations', 'useHeuristic', 'trajectoryIndex', 'timeStep']
        levelValues = [self.numSimulations, self.useHeuristic, trajectoryIndex, timeStep]
        index = pd.MultiIndex.from_product(levelValues, names=levelNames)
        data = [[tup[0], tup[1]] for tup in trajectory]
        df = pd.DataFrame(data = data, index = index, columns = ['State', 'Action'])

        return df


class DirectionalPrior:
    def __init__(self, actionSpace, sheepInitQPos, mostPrefferedAction, sheepId, wolfId, qPosIndex, numQPosEachAgent, priorForMostPreferredAction):
        self.actionSpace = actionSpace
        self.sheepInitQPos = sheepInitQPos
        self.mostPrefferedAction = mostPrefferedAction
        self.sheepId = sheepId
        self.wolfId = wolfId
        self.qPosIndex = qPosIndex
        self.numQPosEachAgent = numQPosEachAgent
        self.priorForMostPrefferedAction = priorForMostPreferredAction

    def __call__(self, currentState):
        sheepCurrentQPos = currentState[self.sheepId][self.qPosIndex:self.qPosIndex+self.numQPosEachAgent]

        if (sheepCurrentQPos == self.sheepInitQPos).all():
            actionPrior = {action: (1-self.priorForMostPrefferedAction)/(len(self.actionSpace)-1) for action in self.actionSpace}
            actionPrior[self.mostPrefferedAction] = self.priorForMostPrefferedAction

        else:
            actionPrior = {action: 1/len(self.actionSpace) for action in self.actionSpace}

        return actionPrior


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transitionFunctionPlay, isTerminal, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transitionFunctionPlay = transitionFunctionPlay
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, allAgentsPolicies):
        worldState = self.reset()

        while self.isTerminal(worldState):
            worldState = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(worldState):
                trajectory.append((worldState, None))
                break
            allAgentsActions = [policy(worldState) for policy in allAgentsPolicies]
            trajectory.append((worldState, allAgentsActions))
            nextState = self.transitionFunctionPlay(worldState, allAgentsActions)
            worldState = nextState

        return trajectory


class Evaluate:
    def __init__(self, measurementFunction, measurementMetricNames):
        self.measurementFunction = measurementFunction
        self.measurementMetricNames = measurementMetricNames

    def __call__(self, modelDf):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)
        numAgent = 2
        envModelName = 'twoAgents'
        numSimulationFrames = 20
        sheepId = 0
        wolfId = 1
        xPosIndex = 2
        numXPosEachAgent = 2
        qVelInit = [0, 0, 0, 0]
        qPosInitNoise = 0
        qVelInitNoise = 0

        levelNames = ['sheepXPosInit', 'sheepYPosInit', 'wolfXPosInit', 'wolfYPosInit', 'cInit', 'cBase', 'numSimulations',
                      'numTrials', 'maxRunningSteps', 'sheepPolicyName', 'render', 'killzoneRadius', 'aliveBonus', 'deathPenalty',
                      'rolloutHeuristicWeight', 'prior']
        levelValues = [modelDf.index.get_level_values(levelName)[0] for levelName in levelNames]
        sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit, cInit, cBase, numSimulations, numTrials, maxRunningSteps, sheepPolicyName, render, killzoneRadius, aliveBonus, deathPenalty, rolloutHeuristicWeight, prior = levelValues

        qPosInitAllAgents = [sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit]

        sheepInitQPos = [sheepXPosInit, sheepYPosInit]
        print('sheepInitQPos', sheepInitQPos)
        mostPrefferedAction = (10, 0)
        priorForMostPrefferedAction = 0.9
        nonUniformPrior = DirectionalPrior(actionSpace, sheepInitQPos, mostPrefferedAction, sheepId, wolfId, xPosIndex, numXPosEachAgent, priorForMostPrefferedAction)
        uniformPrior = GetActionPrior(actionSpace)

        isTerminal = env.IsTerminal(killzoneRadius)

        transitionNoRender = env.TransitionFunction(envModelName, isTerminal, False, numSimulationFrames)
        transitionSheepSimulation = TransitionFunctionSheepSimulation(transitionNoRender, stationaryWolfPolicy)

        transitionPlay = env.TransitionFunction(envModelName, isTerminal, render, numSimulationFrames)
        rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        if sheepPolicyName == 'mcts':
            calculateScore = CalculateScore(cInit, cBase)
            selectChild = SelectChild(calculateScore)

            if prior == 'nonUniform':
                initializeChildren = InitializeChildren(actionSpace, transitionSheepSimulation, nonUniformPrior)
            else:
                initializeChildren = InitializeChildren(actionSpace, transitionSheepSimulation, uniformPrior)

            expand = Expand(isTerminal, initializeChildren)

            rolloutHeuristic = RolloutHeuristicBasedOnClosenessToTarget(rolloutHeuristicWeight, sheepId, wolfId, xPosIndex, numXPosEachAgent)
            rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
            maxRollOutSteps = 10
            rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionSheepSimulation, rewardFunction, isTerminal, rolloutHeuristic)

            selectNextAction = SelectNextAction(transitionSheepSimulation)

            sheepPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)

        else:
            sheepPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]

        wolfPolicy = stationaryWolfPolicy

        allAgentsPolicies = [sheepPolicy, wolfPolicy]

        trajectoriesForAllTrials = []
        dataFramesForAllTrials = []

        reset = env.Reset(envModelName, qPosInitAllAgents, qVelInit, numAgent, qPosInitNoise, qVelInitNoise)

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transitionPlay, isTerminal, reset)

        if rolloutHeuristicWeight == 0:
            useHeuristic = False
        else:
            useHeuristic = True
        convertTrajectoryToDf = ConvertTrajectoryToDf(numSimulations, useHeuristic)

        if prior == 'nonUniform':
            for trial in range(numTrials):
                print("Trial Number: ", trial)
                trajectory = sampleTrajectory(allAgentsPolicies)
                trajectoriesForAllTrials.append(trajectory)
                dfForThisTrajectory = convertTrajectoryToDf(trial, trajectory)
                dataFramesForAllTrials.append(dfForThisTrajectory)

                if render:
                    frames = transitionPlay.frames
                    if len(frames) != 0:
                        print("Generating video")
                        skvideo.io.vwrite("./videos/videoTrial{}.mp4".format(trial), frames)

            combinedDf = pd.concat(dataFramesForAllTrials)
            pickle_out = open("trajectories/NumSim{}UseHeuristic{}Prior{}.pickle".format(numSimulations, useHeuristic, prior),
                              "wb")
            pickle.dump(combinedDf, pickle_out)
            pickle_out.close()

        else:
            convertTrajectoriesDfToListOfTuples = ConvertTrajectoriesDfToListOfTuples(numTrials)
            trajectoriesForAllTrials = convertTrajectoriesDfToListOfTuples("trajectories/NumSim{}UseHeuristic{}.pickle".format(numSimulations, False))

        measurements = self.measurementFunction(trajectoriesForAllTrials)
        measurementSeries = pd.Series({measurementMetricName: measurement for measurementMetricName, measurement in zip(self.measurementMetricNames, measurements)})

        return measurementSeries


def drawPerformanceline(dataDf, axForDraw, xLabel, yLabel, yError):
    plotDf = dataDf.reset_index()

    for key, grp in plotDf.groupby(['prior']):
        grp.plot(ax=axForDraw, kind='line', x=xLabel, y=yLabel, label=key, yerr=yError)


def main():
    startTime = datetime.now()

    sheepXPosInit = [-8]
    sheepYPosInit = [0]
    wolfXPosInit = [8]
    wolfYPosInit = [0]
    cInit = [1]
    cBase = [100]
    numSimulations = [5, 7, 9]#[50, 250, 500]
    numTrials = [3]
    maxRunningSteps = [25]
    sheepPolicyName = ['mcts']
    render = [False]
    killzoneRadius = [0.5]
    aliveBonus = [-0.05]
    deathPenalty = [1]
    rolloutHeuristicWeight = [0]
    prior = ['nonUniform']#['uniform', 'nonUniform']

    levelValues = [sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit, cInit, cBase, numSimulations, numTrials, maxRunningSteps, sheepPolicyName, render, killzoneRadius, aliveBonus, deathPenalty, rolloutHeuristicWeight, prior]
    levelNames = ['sheepXPosInit', 'sheepYPosInit', 'wolfXPosInit', 'wolfYPosInit', 'cInit', 'cBase', 'numSimulations',
                      'numTrials', 'maxRunningSteps', 'sheepPolicyName', 'render', 'killzoneRadius', 'aliveBonus', 'deathPenalty',
                      'rolloutHeuristicWeight', 'prior']

    measurementMetricName1 = 'meanDistance'
    measurementMetricName2 = 'distanceStdDev'
    measurementMetricNames = [measurementMetricName1, measurementMetricName2]
    variableParameter1 = 'numSimulations'
    variableParameter2 = 'prior'

    envModelName = 'twoAgents'
    qPosInit = [sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit]
    qPosInit = [item for sublist in qPosInit for item in sublist]
    qPosInitNoise = 0
    qVelInitNoise = 0
    qVelInit = [0, 0, 0, 0]
    numAgents = 2
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

    evaluate = Evaluate(measurementFunction, measurementMetricNames)

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluate)
    print("modelResultDf: ")
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
    plt.xlim((0, 600))
    plt.show()

    endTime = datetime.now()
    print("Time taken: ", endTime-startTime)


if __name__ == "__main__":
    main()
