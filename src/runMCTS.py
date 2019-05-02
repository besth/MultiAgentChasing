import numpy as np
import pygame as pg
import itertools as it
import math

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw
import agentsMotionSimulation as ag
import env
import reward


class RunMCTS:
    def __init__(self, mcts, maxRunningSteps, isTerminal):
        self.mcts = mcts
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal

    def __call__(self, rootNode):
        # Running
        runningStep = 0
        while runningStep < self.maxRunningSteps:
            currState = list(rootNode.id.values())[0]
            if self.isTerminal(currState):
                break
            nextRoot = self.mcts(rootNode)
            rootNode = nextRoot
            runningStep += 1
        
        # Output number of steps to reach the target.
        print(runningStep)
        return runningStep

def evaluate(cInit, cBase):
    actionSpace = [(10,0),(7,7),(0,10),(-7,7),(-10,0),(-7,-7),(0,-10),(7,-7)]
    numActionSpace = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)
    numStateSpace = 4
   
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([40, 60])
    initWolfPositionNoise = np.array([0, 20])
    sheepPositionReset = ag.SheepPositionReset(initSheepPosition, initSheepPositionNoise)
    wolfPositionReset = ag.WolfPositionReset(initWolfPosition, initWolfPositionNoise)
    
    numOneAgentState = 2
    positionIndex = [0, 1]
    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary) 
    sheepPositionTransition = ag.SheepPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    wolfPositionTransition = ag.WolfPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    
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
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)

    aliveBouns = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns, deathPenalty, isTerminal) 
    
    # Hyper-parameters
    numSimulations = 500
    maxRunningSteps = 40

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transition, getActionPrior)
    expand = Expand(transition, isTerminal, initializeChildren)
    selectNextRoot = SelectNextRoot(initializeChildren)

    # Rollout
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 40
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, rewardFunction, isTerminal)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    rootAction = actionSpace[np.random.choice(range(numActionSpace))]
    numTestingIterations = 100
    episodeLengths = []
    # currState = initState
    for step in range(numTestingIterations):
        state, action = None, None
        initState = transition(state, action)
        optimal = math.ceil((np.sqrt(np.sum(np.power(initState[0:2] - initState[2:4], 2))) - minDistance )/10)
        initActionPrior = getActionPrior(initState)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, action_prior=initActionPrior[rootAction], is_expanded=True)
        rootNode = initializeChildren(rootNode)
        episodeLength = runMCTS(rootNode)
        episodeLengths.append(episodeLength - optimal)

    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength)
    return [meanEpisodeLength]

def main():
    
    cInit = [1]
    cBase = [0.01, 1]
    modelResults = {(np.log10(init), np.log10(base)): evaluate(init, base) for init, base in it.product(cInit, cBase)}

    print("Finished evaluating")
    # Visualize
    independentVariableNames = ['cInit', 'cBase']
    draw(modelResults, independentVariableNames)

if __name__ == "__main__":
    main()
    
