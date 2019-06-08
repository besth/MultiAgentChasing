# Generates benchmark for MCTS with starting positions that are specified in the XML file
# Does not look very good. Maybe I can put it into a unittest framework

import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node
import skvideo

# Local import
from algorithms.mcts import CalculateScore, SelectChild, Expand, RollOut, backup, GetActionPrior, MCTS, InitializeChildren, RolloutHeuristicBasedOnClosenessToTarget, NullRolloutHeuristic, SelectNextRoot
from envMujoco import Reset, TransitionFunctionNaivePredator, IsTerminal
import reward
from runMCTSInMujoco import RunMCTS

def main():
    numTrials = 1
    num_simulations = 250
    maxRunningSteps = 25

    envModelName = 'twoAgents'
    actionSpace = [(10, 0), (-10, 0), (0, 10), (0, -10), (7, 7), (7, -7), (-7, 7), (-7, -7)]
    numActionSpace = len(actionSpace)
    renderOn = True
    minXDis = 0.5
    numSimulationFrames = 20
    isTerminal = IsTerminal(minXDis)
    transitionFunction = TransitionFunctionNaivePredator(envModelName, isTerminal, renderOn, numSimulationFrames)
    aliveBouns = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(aliveBouns, deathPenalty, isTerminal)
    transitionNoRender = TransitionFunctionNaivePredator(envModelName, isTerminal, renderOn=False,
                                                             numSimulationFrames=numSimulationFrames)
    reset = Reset('twoAgents')

    c_init = 1
    c_base = 100
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 10

    calculateScore = CalculateScore(c_init, c_base)
    selectChild = SelectChild(calculateScore)
    getActionPrior = GetActionPrior(actionSpace)
    initializeChildren = InitializeChildren(actionSpace, transitionFunction, getActionPrior)
    expand = Expand(transitionFunction, isTerminal, initializeChildren)
    nullRolloutHeuristic = NullRolloutHeuristic()
    selectNextRoot = SelectNextRoot(transitionFunction)

    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction, rewardFunction, isTerminal, num_simulations, nullRolloutHeuristic)
    mcts = MCTS(num_simulations, selectChild, expand, rollout, backup, selectNextRoot)

    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    rootAction = (0, 0)
    episodeLengths = []

    for step in range(numTrials):
        import datetime
        print("Testing step:", step, datetime.datetime.now())
        state = reset(2)
        print("State after calling reset(): ", state)
        action = (0, 0)
        initState = transitionNoRender(state, action)
        print("initState: ", initState)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
        episodeLength, trajectory = runMCTS(rootNode)
        print("episodeLength: ", episodeLength)

        # Record episode length
        episodeLengths.append(episodeLength)
        print("---------------------------")

    meanEpisodeLength = np.mean(episodeLengths)
    print("meanEpisodeLength: ", meanEpisodeLength)

if __name__ == '__main__':
    main()