import numpy as np
import pandas as pd


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
    return np.asarray((0, 0))


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


def computeDistance(pos1, pos2):
    distance = np.linalg.norm((pos1 - pos2), ord=2)
    return distance


# class DistanceBetweenActualAndOptimalNextPosition:
#     def __init__(self, optimalNextPosition, nextPositionIndex, stateIndexInTuple, xPosIndex, numXPosEachAgent, agentID):
#         self.optimalNextPosition = optimalNextPosition
#         self.nextPositionIndex = nextPositionIndex
#         self.stateIndexInTuple = stateIndexInTuple
#         self.xPosIndex = xPosIndex
#         self.numXPosEachAgent = numXPosEachAgent
#         self.agentID = agentID
#
#     def __call__(self, trajectories):
#         statesAtNextStep = [trajectory[self.nextPositionIndex][self.stateIndexInTuple] for trajectory in trajectories]
#         xPosAtNextStep = [state[self.agentID][self.xPosIndex:self.xPosIndex+self.numXPosEachAgent] for state in statesAtNextStep]
#         distanceForAllTrajectories = [computeDistance(nextPosition, self.optimalNextPosition) for nextPosition in xPosAtNextStep]
#         meanDistance = np.mean(distanceForAllTrajectories)
#         distanceStdDev = np.std(distanceForAllTrajectories)
#
#         return [meanDistance, distanceStdDev]


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, optimalNextPosition, nextPositionIndex, stateIndexInTuple, xPosIndex, numXPosEachAgent, agentID):
        self.optimalNextPosition = optimalNextPosition
        self.nextPositionIndex = nextPositionIndex
        self.stateIndexInTuple = stateIndexInTuple
        self.xPosIndex = xPosIndex
        self.numXPosEachAgent = numXPosEachAgent
        self.agentID = agentID

    def __call__(self, trajectoryDf):
        trajectory = trajectoryDf.values[0][0]
        stateAtNextStep = trajectory[self.nextPositionIndex][self.stateIndexInTuple]
        xPosAtNextStep = stateAtNextStep[self.agentID][self.xPosIndex:self.xPosIndex+self.numXPosEachAgent]
        distance = computeDistance(self.optimalNextPosition, xPosAtNextStep)
        distanceSeries = pd.Series({'distance': distance})

        return distanceSeries