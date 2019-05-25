import os
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
            # optimalNextPosition = list(rootNode.children[0].id.values())[0][0][2:4]
            actualNextPosition = list(nextRoot.id.values())[0][0][2:4]
            targetPosition = (9, 0)
            distance = compute_distance(targetPosition, actualNextPosition)
            # print("next root id", nextRoot.id)

            print(runningStep, distance)
            rootNode = nextRoot
            runningStep += 1
        
        # Output number of steps to reach the target.
        return runningStep, distance


def evaluate(cInit, cBase, numSimulations, maxRunningSteps, numTestingIterations):
    # actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    actionSpace = [(10, 0), (-10, 0), (0, 10), (0, -10)]

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


    aliveBouns = 0.05
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(aliveBouns, deathPenalty, isTerminal)
    reset = env.Reset(envModelName)

    # Hyper-parameters
    numSimulations = numSimulations
    maxRunningSteps = maxRunningSteps

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transitionNoRender, getActionPrior)
    expand = Expand(transitionNoRender, isTerminal, initializeChildren)

    # Rollout
    useHeuristic = False
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 10
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionNoRender, rewardFunction, isTerminal, numSimulations, useHeuristic)

    selectNextRoot = SelectNextRoot(transitionWithRender)
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)
    
    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)

    rootAction = (0, 0)
    numTestingIterations = numTestingIterations
    episodeLengths = []
    distancesToTarget = []
    for step in range(numTestingIterations):
        import datetime
        print("Testing step:", step, datetime.datetime.now())
        state = reset(numAgent)
        action = (0, 0)
        initState = transitionNoRender(state, action)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
        episodeLength, distanceToTarget = runMCTS(rootNode)

        # Record episode length
        episodeLengths.append(episodeLength)

        # Record first step distance
        distancesToTarget.append(distanceToTarget)

        # Generate video
        generateVideo = renderOn
        if generateVideo:
            frames = transitionWithRender.frames
            if len(frames) != 0:
                print("Generating video")
                skvideo.io.vwrite("./video_with_heuristic.mp4", frames)

            # transitionWithRender.frames = []

    meanEpisodeLength = np.mean(episodeLengths)
    meanDistanceToTarget = np.mean(distancesToTarget)
    # print("mean episode length:", meanEpisodeLength)
    # f = open("duration.txt", "a+")
    # if useHeuristic:
    #     f = open("data/small_action_distance_heuristic_sim{}.txt".format(numSimulations), "a+")
    # else:
    #     f = open("data/small_action_distance_no_heuristic_sim{}.txt".format(numSimulations), "a+")
    # print("mean distance to target after running {} steps: ".format(maxRunningSteps), meanDistanceToTarget, file=f)
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
@click.option('--num-simulations', default=500, help='number of simulations each MCTS step runs.')
@click.option('--max-running-steps', default=1, help='maximum number of steps in each episode.')
@click.option('--num-trials', default=100, help='number of testing iterations to run')
def main(num_simulations, max_running_steps, num_trials):
    # create directories to store data
    if not os.path.exists('data/'):
        os.mkdir('data/', mode=0o777)

    cInit = [1]
    cBase = [100]
    modelResults = {(np.log10(init), np.log10(base)): evaluate(init, base, num_simulations, max_running_steps, num_trials) for init, base in it.product(cInit, cBase)}
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
    
