import os
import numpy as np
import itertools as it
import pandas as pd
import pickle

from anytree import AnyNode as Node

import skvideo
import skvideo.io

skvideo.setFFmpegPath("/usr/local/bin")

# Local import
from algorithms.mcts import MCTS, CalculateScore, GetActionPrior, SelectNextAction, SelectChild, Expand, RollOut, backup, \
    InitializeChildren, RolloutHeuristicBasedOnClosenessToTarget
import envMujoco as env
import reward

import click


def compute_distance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))


def evaluateMeanEpisodeLength(trajectoriesAllTrials):
    episodeLengthAllTrials = [len(trajectory) for trajectory in trajectoriesAllTrials]
    meanEpisodeLength = np.mean(episodeLengthAllTrials)

    return meanEpisodeLength


class ConvertTrajectoryToDf:
    def __init__(self, numSimulations, useHeuristic):
        self.numSimulations = [numSimulations]
        self.useHeuristic = [useHeuristic]

    def __call__(self, trajectoryIndex, trajectory):
        trajectoryIndex = [trajectoryIndex]
        timeStep = range(len(trajectory))

        levelNames = ['numSimulations', 'useHeuristic', 'trajectoryIndex', 'timeStep']
        levelValues = [self.numSimulations, self.useHeuristic, trajectoryIndex, timeStep]
        index = pd.MultiIndex.from_product(levelValues, names=levelNames)
        data = [[tup[0], tup[1]] for tup in trajectory]
        df = pd.DataFrame(data = data, index = index, columns = ['State', 'Action'])

        return df


class TransitionFunctionSheepSimulation:
    def __init__(self, transitionFunction, wolfPolicySheepSimulation):
        self.transitionFunction = transitionFunction
        self.wolfPolicySheepSimulation = wolfPolicySheepSimulation

    def __call__(self, worldState, sheepAction):
        wolfAction = self.wolfPolicySheepSimulation(worldState)
        allAgentsActions = [sheepAction, wolfAction]
        nextState = self.transitionFunction(worldState, allAgentsActions)

        return nextState


def stationaryWolfPolicy(worldState):
    return np.asarray([0, 0])


class WolfPolicyForceDirectlyTowardsSheep:
    def __init__(self, wolfId, sheepId, xPosIndex, numXPosEachAgent, actionMagnitude):
        self.wolfId = wolfId
        self.sheepId = sheepId
        self.xPosIndex = xPosIndex
        self.numXPosEachAgent = numXPosEachAgent
        self.actionMagnitude = actionMagnitude

    def __call__(self, worldState):
        sheepXPos = worldState[self.sheepId][self.xPosIndex: self.xPosIndex+self.numXPosEachAgent]
        wolfXPos = worldState[self.wolfId][self.xPosIndex: self.xPosIndex+self.numXPosEachAgent]

        sheepAction = sheepXPos - wolfXPos
        sheepActionNorm = np.sum(np.abs(sheepAction))
        if sheepActionNorm != 0:
            sheepAction = sheepAction/sheepActionNorm
            sheepAction *= self.actionMagnitude

        return sheepAction


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transitionFunctionPlay, isTerminal, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transitionFunctionPlay = transitionFunctionPlay
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, allAgentsPolicies):
        worldState = self.reset()

        while self.isTerminal(worldState):
            worldState = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            allAgentsActions = [policy(worldState) for policy in allAgentsPolicies]
            trajectory.append((worldState, allAgentsActions))
            if self.isTerminal(worldState):
                break
            nextState = self.transitionFunctionPlay(worldState, allAgentsActions)
            worldState = nextState

        return trajectory


class Evaluate:
    def __init__(self, measurementFunction, measurementMetricNames, qVelInit, qPosInitNoise, qVelInitNoise):
        self.measurementFunction = measurementFunction
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.numActionSpace = len(self.actionSpace)
        self.getActionPrior = GetActionPrior(self.actionSpace)
        self.numAgent = 2
        self.envModelName = 'twoAgents'
        self.numSimulationFrames = 20
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = 2
        self.numXPosEachAgent = 2
        self.measurementMetricNames = measurementMetricNames
        self.qVelInit = qVelInit
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise

    def __call__(self, modelDf):
        levelNames = ['sheepXPosInit', 'sheepYPosInit', 'wolfXPosInit', 'wolfYPosInit', 'cInit', 'cBase', 'numSimulations',
                      'numTrials', 'maxRunningSteps', 'sheepPolicyName', 'render', 'killzoneRadius', 'aliveBonus', 'deathPenalty',
                      'rolloutHeuristicWeight']
        levelValues = [modelDf.index.get_level_values(levelName)[0] for levelName in levelNames]
        sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit, cInit, cBase, numSimulations, numTrials, maxRunningSteps, sheepPolicyName, render, killzoneRadius, aliveBonus, deathPenalty, rolloutHeuristicWeight = levelValues

        qPosInit = [sheepXPosInit, sheepYPosInit, wolfXPosInit, wolfYPosInit]

        # Terminal status
        isTerminal = env.IsTerminal(killzoneRadius)

        # Transition
        transitionNoRender = env.TransitionFunction(self.envModelName, isTerminal, False, self.numSimulationFrames)
        transitionSheepSimulation = TransitionFunctionSheepSimulation(transitionNoRender, stationaryWolfPolicy)

        transitionPlay = env.TransitionFunction(self.envModelName, isTerminal, render, self.numSimulationFrames)
        rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

        if sheepPolicyName == 'mcts':
            # MCTS algorithm
            # Select child
            calculateScore = CalculateScore(cInit, cBase)
            selectChild = SelectChild(calculateScore)

            # Expand--Always use transitionNoRender (even when render = True) because we won't need to store frames when we are expanding a node
            initializeChildren = InitializeChildren(self.actionSpace, transitionSheepSimulation, self.getActionPrior)
            expand = Expand(isTerminal, initializeChildren)

            # Rollout--Always use transitionNoRender (even when render = True) because we won't need to store frames while rolling out
            rolloutHeuristic = RolloutHeuristicBasedOnClosenessToTarget(rolloutHeuristicWeight, self.sheepId, self.wolfId, self.xPosIndex, self.numXPosEachAgent)
            rolloutPolicy = lambda state: self.actionSpace[np.random.choice(range(self.numActionSpace))]
            maxRollOutSteps = 10
            rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionSheepSimulation, rewardFunction, isTerminal, rolloutHeuristic)

            # SelectNextAction
            selectNextAction = SelectNextAction(transitionSheepSimulation)

            # MCTS
            sheepPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextAction)

        else:   # follow random policy
            sheepPolicy = lambda state: self.actionSpace[np.random.choice(range(self.numActionSpace))]

        # Wolf Policy
        wolfPolicy = stationaryWolfPolicy

        # Combined policies
        allAgentsPolicies = [sheepPolicy, wolfPolicy]

        trajectoriesForAllTrials = []
        dataFramesForAllTrials = []

        reset = env.Reset(self.envModelName, qPosInit, self.qVelInit, self.numAgent, self.qPosInitNoise, self.qVelInitNoise)

        sampleTrajectory = SampleTrajectory(maxRunningSteps, transitionPlay, isTerminal, reset)

        if rolloutHeuristicWeight == 0:
            useHeuristic = False
        else:
            useHeuristic = True
        convertTrajectoryToDf = ConvertTrajectoryToDf(numSimulations, useHeuristic)

        for trial in range(numTrials):
            print("Trial Number: ", trial)
            trajectory = sampleTrajectory(allAgentsPolicies)
            trajectoriesForAllTrials.append(trajectory)
            dfForThisTrajectory = convertTrajectoryToDf(trial, trajectory)
            dataFramesForAllTrials.append(dfForThisTrajectory)

            if render:
                frames = transitionPlay.frames
                if len(frames) != 0:
                    print("Generating video")
                    skvideo.io.vwrite("./videos/videoTrial{}.mp4".format(trial), frames)


        combinedDf = pd.concat(dataFramesForAllTrials)
        pickle_out = open("trajectories/NumSim{}UseHeuristic{}.pickle".format(numSimulations, useHeuristic), "wb")
        pickle.dump(combinedDf, pickle_out)
        pickle_out.close()

        measurements = self.measurementFunction(trajectoriesForAllTrials)
        measurementSeries = pd.Series({measurementMetricName: measurement for measurementMetricName, measurement in zip(self.measurementMetricNames, measurements)})

        return measurementSeries


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


def evaluateEpisodeLength(qPosInit, qVelInit, cInit, cBase, numSimulations, maxRunningSteps, numTrials, algorithm, render, killzone_radius, aliveBonus, deathPenalty, useHeuristic):
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

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
    transitionNoRender = env.TransitionFunction(envModelName, isTerminal, renderOn=False,
                                                             numSimulationFrames=numSimulationFrames)
    transitionWithRender = env.TransitionFunction(envModelName, isTerminal, renderOn=renderOn,
                                                               numSimulationFrames=numSimulationFrames)
    rewardFunction = reward.RewardFunctionCompete(aliveBonus, deathPenalty, isTerminal)

    # Hyper-parameters
    numSimulations = numSimulations
    maxRunningSteps = maxRunningSteps

    # MCTS algorithm
    # Select child
    calculateScore = CalculateScore(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # expand
    initializeChildren = InitializeChildren(actionSpace, transitionNoRender, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # Rollout
    rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
    maxRollOutSteps = 10
    rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionNoRender, rewardFunction, isTerminal, numSimulations,
                      useHeuristic)

    selectNextRoot = SelectNextRoot(transitionWithRender)
    mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectNextRoot)

    runMCTS = RunMCTS(mcts, maxRunningSteps, isTerminal)
    runRandom = RunRandom(maxRunningSteps, isTerminal, actionSpace, transitionNoRender)

    rootAction = (0, 0)
    numTrials = numTrials
    episodeLengths = []
    distancesToTarget = []

    for step in range(numTrials):
        import datetime
        print("Testing step:", step, datetime.datetime.now())
        reset = env.Reset(envModelName, qPosInit, qVelInit, numAgent)
        state = reset()
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
    episodeLengthStdDev = np.std(episodeLengths)

    return meanEpisodeLength, episodeLengthStdDev


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

        prob = number_terminal / number_total
        probs.append(prob)

    return probs


@click.command()
@click.option('--num-simulations', default=100, help='number of simulations each MCTS step runs.')
@click.option('--max-running-steps', default=25, help='maximum number of steps in each episode.')
@click.option('--num-trials', default=1, help='number of testing iterations to run')
@click.option('--algorithm', default='mcts', help='algorithm to run: mcts or random')
@click.option('--render', default=True, help='whether to render')
@click.option('--killzone-radius', default=0.2,
              help='max distance between the two agents so that they collide with each other')
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
                                                               killzone_radius) for init, base in
                    it.product(cInit, cBase)}
    print("Finished evaluating")


if __name__ == "__main__":
    main()
