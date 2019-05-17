import numpy as np
import itertools as it
import math

from anytree import AnyNode as Node
from anytree import RenderTree

import skvideo
import skvideo.io
skvideo.setFFmpegPath("/usr/local/bin")

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextRoot, SelectChild, Expand, RollOut, backup, InitializeChildren
from visualize import draw
# import agentsMotionSimulation as ag
import envMujoco as env
import reward

from testTree import sampleTrajectory
import click



def compute_distance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class RunMCTS:
    def __init__(self, mcts, maxRunningSteps, isTerminal):
        self.mcts = mcts
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal

    def __call__(self, rootNode):
        # Running
        runningStep = 0
        while runningStep < self.maxRunningSteps:
            print("current running step", runningStep)
            currState = list(rootNode.id.values())[0]

            if self.isTerminal(currState):
                break
            nextRoot = self.mcts(rootNode)

            # Test the optimality of first step.
            # print(rootNode.children)
            # for child in rootNode.children:
            #     if list(child.id.keys())[0] == (10, 0):
            #         # print(child)
            #         optimalNextPosition = list(child.id.values())[0][0][2:4]
            optimalNextPosition = list(rootNode.children[0].id.values())[0][0][2:4]
            actualNextPosition = list(nextRoot.id.values())[0][0][2:4]
            distance = compute_distance(optimalNextPosition, actualNextPosition)
            # print(optimalNextPosition, actualNextPosition, distance)

            rootNode = nextRoot
            runningStep += 1
        
        # Output number of steps to reach the target.
        print(runningStep, distance)
        return runningStep, distance

#@click.command()
#@click.option('--numSimulations', default=200, help='number of simulations each MCTS step runs.')
#@click.argument('cInit')
#@click.argument('cBase')
def evaluate(cInit, cBase, numSimulations):
    actionSpace = [(10,0),(7,7),(0,10),(-7,7),(-10,0),(-7,-7),(0,-10),(7,-7)]
    numActionSpace = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)
    numStateSpace = 4
    numAgent = 2

    # Terminal status
    minXDis = 0.2
    isTerminal = env.IsTerminal(minXDis)
   
    # Transition
    envModelName = 'twoAgents'
    renderOn = False
    numSimulationFrames = 20
    transitionNoRender = env.TransitionFunctionNaivePredator(envModelName, isTerminal, renderOn=False, numSimulationFrames=numSimulationFrames)
    transitionWithRender = env.TransitionFunctionNaivePredator(envModelName, isTerminal, renderOn=renderOn, numSimulationFrames=numSimulationFrames)


    aliveBouns = -0.05
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(aliveBouns, deathPenalty, isTerminal)
    reset = env.Reset(envModelName)

    # Hyper-parameters
    numSimulations = numSimulations
    maxRunningSteps = 1

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transitionNoRender, getActionPrior)
    expand = Expand(transitionNoRender, isTerminal, initializeChildren)

    # Rollout
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 20
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionNoRender, rewardFunction, isTerminal, numSimulations)

    selectNextRoot = SelectNextRoot(transitionWithRender)
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    rootAction = (0, 0)
    numTestingIterations = 50
    episodeLengths = []
    firstStepDistances = []
    for step in range(numTestingIterations):
        import datetime
        print("Testing step:", step, datetime.datetime.now())
        state = reset(numAgent)
        action = (0, 0)
        initState = transitionNoRender(state, action)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
        episodeLength, firstStepDistance = runMCTS(rootNode)
        # print(RenderTree(rootNode))

        # Record episode length
        episodeLengths.append(episodeLength)

        # Record first step distance
        firstStepDistances.append(firstStepDistance)

        # Generate video
        generateVideo = renderOn
        if generateVideo:
            frames = transitionWithRender.frames
            if len(frames) != 0:
                print("Generating video")
                skvideo.io.vwrite("./video_with_heuristic.mp4", frames)

            # transitionWithRender.frames = []

    meanEpisodeLength = np.mean(episodeLengths)
    meanFirstStepDistance = np.mean(firstStepDistances)
    # print("mean episode length:", meanEpisodeLength)
    # f = open("duration.txt", "a+")
    # f = open("distance.txt", "a+")
    # print("mean episode length:", meanEpisodeLength, "simulation number:", numSimulations, file=f)
    # print("survival rate:", episodeLengths.count(maxRunningSteps) / len(episodeLengths))

    return [meanEpisodeLength]


# helper function to calculate some test results
def calc_rollout_terminal_prob(distances, num_simulations):
    probs = []
    for dis in distances:
        with open("rollout_total_heuristic_{}_{}.txt".format(dis, num_simulations)) as f1:
            for i, l in enumerate(f1):
                pass
            number_total = i + 1

        with open("rollout_terminal_heuristic_{}_{}.txt".format(dis, num_simulations)) as f2:
            for i, l in enumerate(f2):
                pass
            number_terminal = i + 1

        prob = number_terminal/number_total
        probs.append(prob)
    return probs


@click.command()
@click.option('--num-simulations', default=200, help='number of simulations each MCTS step runs.')
def main(num_simulations):

    cInit = [1]
    cBase = [100]
    modelResults = {(np.log10(init), np.log10(base)): evaluate(init, base, num_simulations) for init, base in it.product(cInit, cBase)}
    print("Finished evaluating")

    # Visualize
    # independentVariableNames = ['cInit', 'cBase']
    # draw(modelResults, independentVariableNames)

    # Generate test results.
    # distances = [1, 2, 4, 8, 12, 16]
    # distances = [16]
    # probs = calc_rollout_terminal_prob(distances, num_simulations)
    # f = open("rollout_prob_heuristic.txt", "a+")
    # print("prob", probs, "number of simulations", num_simulations, file=f)

if __name__ == "__main__":
    main()
    
