import numpy as np
import copy
import itertools as it

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, ResetRoot
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
    actionSpace = [-1, 1]
    numActions = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)

    # Reward function
    stepPenalty = -0.1
    catchReward = 1
    targetState = envBoundHigh
    isTerminal = Terminal(targetState)
    reward = RewardFunction(stepPenalty, catchReward, isTerminal)

    # UCB score calculation - values from AlphaZero source code
    calculateScore = CalculateScore(cInit, cBase)

    selectChild = SelectChild(calculateScore)
    expand = Expand(getActionPrior, transition, isTerminal)

    # Rollout
    rolloutPolicy = lambda state: np.random.choice(actionSpace)
    maxRollOutSteps = 10
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, reward, isTerminal)

    # Hyper-parameters
    numSimulations = 1000
    maxRunningSteps = 20

    # MCTS algorithm
    resetRoot = ResetRoot(actionSpace, transition, getActionPrior)
    selectNextRoot = SelectNextRoot(resetRoot)
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)

    initState = 0
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    # testing
    numTestingIterations = 100
    episodeLengths = []
    # currState = initState
    for step in range(numTestingIterations):
        rootNode = resetRoot(initState)
        episodeLength = runMCTS(rootNode)
        # rootNode = Node(id={rootAction: initState}, num_visited=1, sum_value=0, action_prior=initActionPrior[rootAction], is_expanded=True)
        episodeLengths.append(episodeLength)

    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength)
    return [meanEpisodeLength]

def main():
    
    cInit = [1, 10, 100, 1000, 10000]
    cBase = [1, 10, 100, 1000, 10000]
    modelResults = {(init, base): evaluate(init, base) for init, base in it.product(cInit, cBase)}

    print("Finished evaluating")
    # Visualize
    independentVariableNames = ['cInit', 'cBase']
    draw(modelResults, independentVariableNames)

if __name__ == "__main__":
    main()
    
