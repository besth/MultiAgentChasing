import numpy as np
import pygame as pg
import itertools as it
import math

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
# from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
from algorithms.stochasticMCTS import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren

from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw
import stochasticAgentsMotionSimulation as ag
import env
import reward


class RunMCTS:
    def __init__(self, mcts, maxRunningSteps, isTerminal, render):
        self.mcts = mcts
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.render = render

    def __call__(self, rootNode):
        # Running
        runningStep = 0
        while runningStep < self.maxRunningSteps:
            currState = list(rootNode.id.values())[0]
            self.render(currState)
            if self.isTerminal(currState):
                break
            nextRoot = self.mcts(rootNode)
            rootNode = nextRoot
            runningStep += 1
        
        # Output number of steps to reach the target.
        print(runningStep)
        return runningStep

def evaluate(numTree, chasingSubtlety, cInit = 1, cBase = 10):
    actionSpace = [(10,0),(7,7),(0,10),(-7,7),(-10,0),(-7,-7),(0,-10),(7,-7)]
    numActionSpace = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)
    numStateSpace = 4

    # 2D Env
    initSheepPosition = np.array([60, 60]) 
    initWolfPosition = np.array([40, 40])
    initSheepPositionNoise = np.array([0, 0])
    initWolfPositionNoise = np.array([0, 0])
    sheepPositionReset = ag.SheepPositionReset(initSheepPosition, initSheepPositionNoise)
    wolfPositionReset = ag.WolfPositionReset(initWolfPosition, initWolfPositionNoise)
    
    numOneAgentState = 2
    positionIndex = [0, 1]
    xBoundary = [0, 80]
    yBoundary = [0, 80]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary) 
    sheepPositionTransition = ag.SheepPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    wolfSpeed = 7
    #chasingSubtlety = 3.3
    wolfPositionTransition = ag.WolfPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust, wolfSpeed, chasingSubtlety) 
    
    numAgent = 2
    sheepId = 0
    wolfId = 1
    transition = env.TransitionFunction(sheepId, wolfId, sheepPositionReset, wolfPositionReset, sheepPositionTransition, wolfPositionTransition)
    minDistance = 10
    isTerminal = env.IsTerminal(sheepId, wolfId, numOneAgentState, positionIndex, minDistance) 
     
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255,255,255]
    circleColorList = [[50,255,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50]]
    circleSize = 8
    saveImage = True
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)

    aliveBouns = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns, deathPenalty, isTerminal) 
    
    # Hyper-parameters
    numSimulations = int(2048/numTree)
    maxRunningSteps = 20

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
    expand = Expand(transition, isTerminal, initializeChildren)
    selectNextRoot = SelectNextRoot(transition)

    # Rollout
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 60
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)

    #numTree = 10
    mcts = MCTS(numTree, numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal, render)

    rootAction = actionSpace[np.random.choice(range(numActionSpace))]
    numTestingIterations = 30
    episodeLengths = []
    for step in range(numTestingIterations):
        import datetime
        print (datetime.datetime.now())
        state, action = None, None
        initState = transition(state, action)
        #optimal = math.ceil((np.sqrt(np.sum(np.power(initState[0:2] - initState[2:4], 2))) - minDistance )/10)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded = False)
        episodeLength = runMCTS(rootNode)
        episodeLengths.append(episodeLength)
    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength)
    return [meanEpisodeLength]

def evaluate1D(cInit, cBase):
    # Transition function
    envBoundLow = 0
    envBoundHigh = 7
    transition = TransitionFunction(envBoundLow, envBoundHigh)
    # Action space
    actionSpace = [-1,1]
    numActions = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)
    # Reward function
    stepPenalty = -0.05
    catchReward = 1
    targetState = envBoundHigh
    isTerminal = Terminal(targetState)
    reward = RewardFunction(stepPenalty, catchReward, isTerminal)

    # Hyper-parameters
    numSimulations = 100
    maxRunningSteps = 20

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
    expand = Expand(transition, isTerminal, initializeChildren)
    selectNextRoot = SelectNextRoot(initializeChildren, transition)

    # Rollout
    rolloutPolicy = lambda state: np.random.choice(actionSpace)
    maxRollOutSteps = 60
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, reward, isTerminal)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)

    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

def main():
    
    #cInit = [1]
    #cBase = [1]
    #modelResults = {(np.log10(init), np.log10(base)): evaluate(init, base) for init, base in it.product(cInit, cBase)}

    numTrees = [1,2]
    subtleties = [50,1.83]
    modelResults = {(numTree, chasingSubtlety): evaluate(numTree, chasingSubtlety) for numTree, chasingSubtlety in it.product(numTrees, subtleties)}

    print("Finished evaluating")
    # Visualize
    independentVariableNames = ['cInit', 'cBase']
    independentVariableNames = ['numTree', 'chasingSubtlety']
    draw(modelResults, independentVariableNames)

if __name__ == "__main__":
    main()
    
