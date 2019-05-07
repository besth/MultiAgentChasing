import numpy as np
import itertools as it
import math

from anytree import AnyNode as Node
from anytree import RenderTree

import skvideo.io

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
from visualize import draw
# import agentsMotionSimulation as ag
import envMujoco as env
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
    numAgent = 2
   
    envModelName = 'twoAgents'
    renderOn = True
    transitionNoRender = env.TransitionFunctionNaivePredator(envModelName, renderOn=False)
    transitionWithRender = env.TransitionFunctionNaivePredator(envModelName, renderOn=renderOn)

    minXDis = 0.2
    isTerminal = env.IsTerminal(minXDis)

    aliveBouns = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(aliveBouns, deathPenalty, isTerminal)
    reset = env.Reset(envModelName)

    # Hyper-parameters
    numSimulations = 200
    maxRunningSteps = 200

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transitionNoRender, getActionPrior)
    expand = Expand(transitionNoRender, isTerminal, initializeChildren)

    # Rollout
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 100
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionNoRender, rewardFunction, isTerminal)

    selectNextRoot = SelectNextRoot(transitionWithRender)
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    # rootAction = actionSpace[np.random.choice(range(numActionSpace))]
    rootAction = (0, 0)
    numTestingIterations = 10
    episodeLengths = []
    for step in range(numTestingIterations):
        import datetime
        print("Testing step:", step, datetime.datetime.now())
        state = reset(numAgent)
        action = (0, 0)
        initState = transitionNoRender(state, action)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
        episodeLength = runMCTS(rootNode)
        episodeLengths.append(episodeLength)

        # Generate video
        frames = transitionWithRender.frames
        if len(frames) != 0:
            print("Generating video")
            skvideo.io.vwrite("./video.mp4", frames)

    meanEpisodeLength = np.mean(episodeLengths)
    print("mean episode length is", meanEpisodeLength)




    return [meanEpisodeLength]

def main():
    
    # cInit = [0.1, 1, 10]
    # cBase = [0.01, 0.1, 1]
    cInit = [0.1]
    cBase = [0.1]
    modelResults = {(np.log10(init), np.log10(base)): evaluate(init, base) for init, base in it.product(cInit, cBase)}

    print("Finished evaluating")
    # Visualize
    independentVariableNames = ['cInit', 'cBase']
    # draw(modelResults, independentVariableNames)

if __name__ == "__main__":
    main()
    
