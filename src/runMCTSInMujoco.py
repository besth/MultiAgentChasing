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
            print("(MCTS) current running step", runningStep)
            currState = list(rootNode.id.values())[0]

            if self.isTerminal(currState):
                break
            nextRoot = self.mcts(rootNode)

            print(runningStep)
            rootNode = nextRoot
            runningStep += 1
        
        # Output number of steps to reach the target.
        return runningStep


class RunRandom:
    def __init__(self, maxRunningSteps, isTerminal, actionSpace, transition_func):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.actionSpace = actionSpace
        self.transition_func = transition_func

    def __call__(self, initState):
        # Running
        runningStep = 0
        currState = initState
        while runningStep < self.maxRunningSteps:
            print("(Random) current running step", runningStep)

            if self.isTerminal(currState):
                break

            actionIndex = np.random.choice(range(len(self.actionSpace)))
            action = self.actionSpace[actionIndex]
            nextState = self.transition_func(currState, action)

            currState = nextState
            runningStep += 1

        # Output number of steps to reach the target.
        return runningStep


def evaluate(cInit, cBase, numSimulations, maxRunningSteps, numTestingIterations, algorithm, render, killzone_radius):
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    # actionSpace = [(10, 0), (-10, 0), (0, 10), (0, -10)]

    numActionSpace = len(actionSpace)
    getActionPrior = GetActionPrior(actionSpace)
    numStateSpace = 4
    numAgent = 2

    # Terminal status
    minXDis = killzone_radius
    isTerminal = env.IsTerminal(minXDis)
   
    # Transition
    envModelName = 'twoAgents'
    renderOn = render
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
    runRandom = RunRandom(maxRunningSteps, isTerminal, actionSpace, transitionNoRender)

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

        if algorithm == "mcts":
            rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
            episodeLength = runMCTS(rootNode)
        else:
            episodeLength = runRandom(initState)

        # Record episode length
        episodeLengths.append(episodeLength)

        # Generate video
        generateVideo = renderOn
        if generateVideo:
            frames = transitionWithRender.frames
            if len(frames) != 0:
                print("Generating video")
                skvideo.io.vwrite("./video.mp4", frames)

    meanEpisodeLength = np.mean(episodeLengths)
    dis = 9
    f = open("data/corner_mean_episode_length_{}_sim{}".format(algorithm, numSimulations), "a+")
    print("Mean episode length at distance {} is: {}".format(dis, meanEpisodeLength), file=f)
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
@click.option('--num-simulations', default=250, help='number of simulations each MCTS step runs.')
@click.option('--max-running-steps', default=100, help='maximum number of steps in each episode.')
@click.option('--num-trials', default=50, help='number of testing iterations to run')
@click.option('--algorithm', default='mcts', help='algorithm to run: mcts or random')
@click.option('--render', default=False, help='whether to render')
@click.option('--killzone-radius', default=0.2, help='max distance between the two agents so that they collide with each other')
def main(num_simulations, max_running_steps, num_trials, algorithm, render, killzone_radius):
    # create directories to store data
    if not os.path.exists('data/'):
        os.mkdir('data/', mode=0o777)

    cInit = [1]
    cBase = [100]
    modelResults = {(np.log10(init), np.log10(base)): evaluate(init,
                                                               base,
                                                               num_simulations,
                                                               max_running_steps,
                                                               num_trials,
                                                               algorithm,
                                                               render,
                                                               killzone_radius) for init, base in it.product(cInit, cBase)}
    print("Finished evaluating")


if __name__ == "__main__":
    main()
    
