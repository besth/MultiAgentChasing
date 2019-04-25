import numpy as np

from anytree import AnyNode as Node
from anytree import RenderTree

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, select_next_root, SelectChild, Expand, RollOut, backup
from simple1DEnv import TransitionFunction, RewardFunction, Terminal

def simulate():
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
    maxRollOutSteps = 100
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transition, reward, isTerminal)

    # Hyper-parameters
    numSimulations = 50
    maxRunningSteps = 200

    # MCTS algorithm
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, select_next_root)

    initState = 5
    rootAction = np.random.choice(actionSpace)
    initActionPrior = getActionPrior(initState)
    rootNode = Node(id={rootAction: initState}, num_visited=1, sum_value=0, action_prior=initActionPrior[rootAction], is_expanded=True)

    # Create non-expanded children of the root.
    for action in actionSpace:
        nextState = transition(initState, action)
        Node(parent=rootNode, id={action: nextState}, num_visited=1, sum_value=0, action_prior=initActionPrior[action], is_expanded=False)
    
    # Running
    currState = initState
    runningStep = 0
    while runningStep < maxRunningSteps:
        if isTerminal(currState):
            break
        nextRoot = mcts(rootNode)
        action = list(nextRoot.id.keys())[0]
        nextState = transition(currState, action)

        currState = nextState
        rootNode = nextRoot
        runningStep += 1
    
    # Output number of steps to reach the target.
    return runningStep


if __name__ == "__main__":
    episodeLength = []
    for i in range(100):
        length = simulate()
        episodeLength.append(length)
        print(i, length)
    
    print(np.mean(episodeLength))
    # print(episodeLength.count(2))