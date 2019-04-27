import numpy as np
import copy
import itertools as it

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw


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
    selectNextRoot = SelectNextRoot(initializeChildren)

    # Rollout
    rolloutPolicy = lambda state: np.random.choice(actionSpace)
    maxRollOutSteps = 60
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, reward, isTerminal)

    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    # testing
    initState = 0
    initActionPrior = getActionPrior(initState)
    rootAction = np.random.choice(actionSpace)
    numTestingIterations = 100
    episodeLengths = []
    # currState = initState
    for step in range(numTestingIterations):
        rootNode = Node(id={rootAction: initState}, num_visited=1, sum_value=0, action_prior=initActionPrior[rootAction], is_expanded=True)
        rootNode = initializeChildren(rootNode)
        episodeLength = runMCTS(rootNode)
        episodeLengths.append(episodeLength)

    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength)
    return [meanEpisodeLength]

def main():
    
    cInit = [1/10000, 1/100, 1, 100, 10000]
    cBase = [100, 1000]
    modelResults = {(init, base): evaluate(init, base) for init, base in it.product(cInit, cBase)}

    print("Finished evaluating")
    # Visualize
    independentVariableNames = ['cInit', 'cBase']
    draw(modelResults, independentVariableNames)

if __name__ == "__main__":
    main()
    
