import numpy as np
import copy

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, ResetRoot
from simple1DEnv import TransitionFunction, RewardFunction, Terminal


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

def main():
    # Transition function
    envBoundLow = 0
    envBoundHigh = 7
    transition = TransitionFunction(envBoundLow, envBoundHigh)

    # Action space
    actionSpace = [-1, 1]
    numActions = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)

    # Reward function
    stepPenalty = -1
    catchReward = 1
    targetState = envBoundHigh
    isTerminal = Terminal(targetState)
    reward = RewardFunction(stepPenalty, catchReward, isTerminal)

    # UCB score calculation - values from AlphaZero source code
    cInit = 1.25
    cBase = 19652
    calculateScore = CalculateScore(cInit, cBase)

    selectChild = SelectChild(calculateScore)
    expand = Expand(getActionPrior, transition, isTerminal)

    # Rollout
    rolloutPolicy = lambda state: np.random.choice(actionSpace)
    maxRollOutSteps = 200
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, reward, isTerminal)

    # Hyper-parameters
    numSimulations = 10000
    maxRunningSteps = 200

    # MCTS algorithm
    resetRoot = ResetRoot(actionSpace, transition, getActionPrior)
    selectNextRoot = SelectNextRoot(resetRoot)
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)

    initState = 0
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    # testing
    numTestingIterations = 10
    episodeLengths = []
    # currState = initState
    for step in range(numTestingIterations):
        rootNode = resetRoot(initState)
        print("Curr step", step)
        episodeLength = runMCTS(rootNode)
        # rootNode = Node(id={rootAction: initState}, num_visited=1, sum_value=0, action_prior=initActionPrior[rootAction], is_expanded=True)
        episodeLengths.append(episodeLength)

    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength)

if __name__ == "__main__":
    main()
    
