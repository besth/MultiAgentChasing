import pandas as pd
import pylab as plt
import numpy as np
from anytree import AnyNode as Node
from matplotlib import pyplot as plt

# Local import
from algorithms.mcts import CalculateScore, SelectChild, Expand, RollOut, backup, GetActionPrior, MCTS, InitializeChildren, SelectNextRoot
from envMujoco import Reset, TransitionFunction, IsTerminal
import reward
from runMCTSInMujoco import evaluateEpisodeLength


def distanceBetweenActualAndOptimalNextPosition(mcts, rootNode, numTrials, optimalNextPositionOfPrey):
    L2NormsEachTrial = []

    for trial in range(numTrials):
        nextRoot = mcts(rootNode)
        nextPositionOfPrey = list(nextRoot.id.values())[0][0][2:4]

        # L2 norm between actual and optimal next step
        L2Norm = np.linalg.norm((optimalNextPositionOfPrey - nextPositionOfPrey), ord=2)
        L2NormsEachTrial.append(L2Norm)

    meanL2Norm = np.mean(np.array(L2NormsEachTrial))
    return meanL2Norm


# def testOptimalityOfFirstStep(modelDf, evaluationMetricName):   # Might require some changes
#     parameters = modelDf.reset_index()
#
#     cInit = parameters['cInit'][0]
#     cBase = parameters['cBase'][0]
#     numSimulations = parameters['numSimulations'][0]
#     maxRunningSteps = parameters['maxRunningSteps'][0]
#     algorithm = parameters['algorithm'][0]
#     render = parameters['render'][0]
#     killzone_radius = parameters['killzone_radius'][0]
#     numTrials = parameters['numTrials'][0]
#
#     envModelName = 'twoAgents'
#     actionSpace = [(10, 0), (-10, 0), (0, 10), (0, -10), (7, 7), (7, -7), (-7, 7), (-7, -7)]
#     numActionSpace = len(actionSpace)
#     numSimulationFrames = 20
#     isTerminal = IsTerminal(killzone_radius)
#     transitionFunction = TransitionFunction(envModelName, isTerminal, render,
#                                                               numSimulationFrames)
#     aliveBouns = -0.05
#     deathPenalty = 1
#     rewardFunction = reward.RewardFunctionCompete(aliveBouns, deathPenalty, isTerminal)
#     reset = Reset(envModelName, 2)
#
#     rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
#     maxRollOutSteps = 10
#     rollout_heuristic = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction,
#                                      rewardFunction, isTerminal,
#                                      numSimulations, True)
#     # rollout_no_heuristic = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction,
#     #                                     rewardFunction, isTerminal,
#     #                                     numSimulations, False)
#
#     calculateScore = CalculateScore(cInit, cBase)
#     selectChild = SelectChild(calculateScore)
#     getActionPrior = GetActionPrior(actionSpace)
#     initializeChildren = InitializeChildren(actionSpace, transitionFunction, getActionPrior)
#     expand = Expand(isTerminal, initializeChildren)
#     selectNextRoot = SelectNextRoot(transitionFunction)
#
#     rootAction = (0, 0)
#     initState = reset(2)
#     print("Initial State: ", initState)
#     rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
#     optimalAction = (10, 0)
#     optimalNextState = transitionFunction(initState, optimalAction)
#     optimalPositionOfPreyAfterOneStep = optimalNextState[0][2:4]
#     print("Optimal Next State: ", optimalNextState)
#
#     mcts_heuristic = MCTS(numSimulations, selectChild, expand, rollout_heuristic, backup,
#                           selectNextRoot)
#     # mcts_no_heuristic = MCTS(numSimulations, selectChild, expand, rollout_no_heuristic, backup,
#     #                          selectNextRoot)
#     meanL2Norm_Heuristic = distanceBetweenActualAndOptimalNextPosition(mcts_heuristic, rootNode, numTrials,
#                                                                             optimalPositionOfPreyAfterOneStep)
#     # meanL2Norm_NoHeuristic = distanceBetweenActualAndOptimalNextPosition(mcts_no_heuristic, rootNode,
#     #                                                                           numTrials,
#     #                                                                           optimalPositionOfPreyAfterOneStep)
#
#     resultSe = pd.Series({evaluationMetricName: meanL2Norm_Heuristic})
#     return resultSe


def computeEpisodeLength(modelDf, evaluationMetricName1, evaluationMetricName2, xLabel):
    parameters = modelDf.reset_index()

    cInit = parameters['cInit'][0]
    cBase = parameters['cBase'][0]
    numSimulations = parameters['numSimulations'][0]
    maxRunningSteps = parameters['maxRunningSteps'][0]
    algorithm = parameters['algorithm'][0]
    render = parameters['render'][0]
    killzoneRadius = parameters['killzoneRadius'][0]
    numTrials = parameters['numTrials'][0]
    deathPenalty = parameters['deathPenalty'][0]
    aliveBonus = parameters['aliveBonus'][0]
    envAgentInitPosition = parameters['envAgentInitPosition'][0]
    initPosition = parameters['initPosition'][0]
    useHeuristic = parameters['useHeuristic'][0]
    qPosInit = [initPosition, 0, envAgentInitPosition, 0]
    qVelInit = [0, 0, 0, 0]

    meanEpisodeLength, episodeLengthStdDev = evaluateEpisodeLength(qPosInit, qVelInit, cInit, cBase, numSimulations, maxRunningSteps, numTrials, algorithm, render, killzoneRadius, aliveBonus, deathPenalty, useHeuristic)
    resultSe = pd.Series({evaluationMetricName1: meanEpisodeLength, evaluationMetricName2: episodeLengthStdDev, xLabel: envAgentInitPosition-initPosition})

    return resultSe


def drawHeatmap(dataDf, axForDraw, xLabel, yLabel, c):
    plotDf = dataDf.reset_index()
    plotDf.plot.scatter(x=xLabel, y=yLabel, c=c, colormap="viridis", ax=axForDraw)


def drawPerformanceline(dataDf, axForDraw, xLabel, yLabel, yError, variableParameter2):
    plotDf = dataDf.reset_index()

    DfUseHeuristicTrue = plotDf.loc[plotDf[variableParameter2] == True]
    DfUseHeuristicTrue.plot(x=xLabel, y=yLabel, ax=axForDraw, yerr=yError, capsize=4)

    DfUseHeuristicFalse = plotDf.loc[plotDf[variableParameter2] == False]
    DfUseHeuristicFalse.plot(x=xLabel, y=yLabel, ax=axForDraw, yerr=yError, capsize=4)


def main():
    numTrials = [1]#[150]
    render = [False]
    algorithm = ['mcts']
    cInit = [1]
    cBase = [100]
    maxRunningSteps = [5]#[25]
    killzoneRadius = [0.5]
    numSimulations = [1]#[200]
    aliveBonus = [-0.05]
    deathPenalty = [1]
    envAgentInitPosition = [9]
    initPosition = [5, 1, -3, -7]
    useHeuristic = [True, False]

    levelValues = [initPosition, envAgentInitPosition, numTrials, killzoneRadius, numSimulations, render, algorithm, cInit, cBase, maxRunningSteps, aliveBonus, deathPenalty, useHeuristic]
    levelNames = ["initPosition", "envAgentInitPosition", "numTrials", "killzoneRadius", "numSimulations", "render", "algorithm", "cInit", "cBase", "maxRunningSteps", "aliveBonus", "deathPenalty", "useHeuristic"]

    evaluationMetricName1 = 'meanEpisodeLength'
    evaluationMetricName2 = 'episodeLengthStdDev'
    variableParameter1 = 'initPosition'
    variableParameter2 = 'useHeuristic'
    xLabel = 'distance'

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    toSplitFrame = pd.DataFrame(index=modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(computeEpisodeLength, evaluationMetricName1, evaluationMetricName2, xLabel)
    print(modelResultDf)

    fig = plt.figure()

    plotRowNum = 1
    plotColNum = 1
    plotCounter = 1

    levelNames.remove(variableParameter1)
    levelNames.remove(variableParameter2)

    for (key, dataDf) in modelResultDf.groupby(levelNames):
        axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
        drawPerformanceline(dataDf, axForDraw, xLabel, evaluationMetricName1, evaluationMetricName2, variableParameter2)

    plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
