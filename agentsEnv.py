import numpy as np 
import AnalyticGeometryFunctions as ag

class SheepPositionAndVelocityReset():
    def __init__(self, initSheepPosition, initSheepVelocity, initSheepPositionNoise, checkBoundaryAndAdjust):
        self.initSheepPosition = initSheepPosition
        self.initSheepVelocity = initSheepVelocity
        self.initSheepPositionNoise = initSheepPositionNoise
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self):
        startSheepPosition = self.initSheepPosition + np.random.uniform(-self.initSheepPositionNoise, self.initSheepPositionNoise)
        startSheepVelocity = self.initSheepVelocity
        checkedPosition, checkedVelocity, toWallDistance = self.checkBoundaryAndAdjust(startSheepPosition, startSheepVelocity)
        startSheepPositionAndVelocity = np.concatenate([checkedPosition, checkedVelocity, toWallDistance])
        return startSheepPositionAndVelocity

class WolfPositionAndVelocityReset():
    def __init__(self, initWolfPosition, initWolfVelocity, initWolfPositionNoise, checkBoundaryAndAdjust):
        self.initWolfPosition = initWolfPosition
        self.initWolfVelocity = initWolfVelocity
        self.initWolfPositionNoise = initWolfPositionNoise
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self):
        startWolfPosition = self.initWolfPosition + np.random.uniform(-self.initWolfPositionNoise, self.initWolfPositionNoise)
        startWolfVelocity = self.initWolfVelocity
        checkedPosition, checkedVelocity, toWallDistance = self.checkBoundaryAndAdjust(startWolfPosition, startWolfVelocity)
        startWolfPositionAndVelocity = np.concatenate([checkedPosition, checkedVelocity, toWallDistance])
        return startWolfPositionAndVelocity

class SheepPositionAndVelocityTransation():
    def __init__(self, sheepVelocitySpeed, sheepActionFrequency, numOneAgentState, positionIndex, velocityIndex, checkBoundaryAndAdjust):
        self.sheepVelocitySpeed = sheepVelocitySpeed
        self.sheepActionFrequency = sheepActionFrequency
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.velocityIndex = velocityIndex
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self, oldAllAgentState, sheepId, sheepAction, timeStep):
        oldSheepState = oldAllAgentState[self.numOneAgentState * sheepId : self.numOneAgentState * (sheepId + 1)]
        oldSheepPosition = oldSheepState[min(self.positionIndex) : max(self.positionIndex) + 1]
        oldSheepVelocity = oldSheepState[min(self.velocityIndex) : max(self.velocityIndex) + 1]
     
        if timeStep % self.sheepActionFrequency == 0:
            newSheepVelocity = sheepAction
        else:
            newSheepVelocity = oldSheepVelocity

        newSheepPosition = oldSheepPosition + newSheepVelocity
        checkedPosition, checkedVelocity, toWallDistance = self.checkBoundaryAndAdjust(newSheepPosition, newSheepVelocity)
        sheepPositionAndVelocity = np.concatenate([checkedPosition, checkedVelocity, toWallDistance])
        return sheepPositionAndVelocity

class WolfPositionAndVelocityTransation():
    def __init__(self, wolfVelocitySpeed, wolfActionFrequency, numOneAgentState, positionIndex, velocityIndex, checkBoundaryAndAdjust):
        self.wolfVelocitySpeed = wolfVelocitySpeed
        self.wolfActionFrequency = wolfActionFrequency
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.velocityIndex = velocityIndex
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self, oldAllAgentState, sheepId, wolfId, wolfSubtlety, timeStep):
        oldWolfState = oldAllAgentState[self.numOneAgentState * wolfId : self.numOneAgentState * (wolfId + 1)]
        oldWolfPosition = oldWolfState[min(self.positionIndex) : max(self.positionIndex) + 1]
        oldWolfVelocity = oldWolfState[min(self.velocityIndex) : max(self.velocityIndex) + 1]
        
        
        if timeStep % self.wolfActionFrequency == 0:
            oldSheepState = oldAllAgentState[self.numOneAgentState * sheepId : self.numOneAgentState * (sheepId + 1)]
            oldSheepPosition = oldSheepState[min(self.positionIndex) : max(self.positionIndex) + 1]
            heatSeekingDirectionCartesian = oldSheepPosition - oldWolfPosition
            heatSeekingDirectionPolar = ag.transiteCartesianToPolar(heatSeekingDirectionCartesian)
            wolfVelocityDirectionPolar = np.random.vonmises(heatSeekingDirectionPolar, wolfSubtlety)
            wolfVelocityDirectionCartesian = ag.transitePolarToCartesian(wolfVelocityDirectionPolar)
            newWolfVelocity = wolfVelocityDirectionCartesian * self.wolfVelocitySpeed
        else:
            newWolfVelocity = oldWolfVelocity

        newWolfPosition = oldWolfPosition + newWolfVelocity
        
        checkedPosition, checkedVelocity, toWallDistance = self.checkBoundaryAndAdjust(newWolfPosition, newWolfVelocity)
        wolfPositionAndVelocity = np.concatenate([checkedPosition, checkedVelocity, toWallDistance])

        return wolfPositionAndVelocity

class CheckBoundaryAndAdjust():
    def __init__(self, xBoundary, yBoundary):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
    def __call__(self, position, velocity):
        if position[0] >= self.xMax:
#            position[0] = self.xMax
            position[0] = 2 * self.xMax - position[0]
            velocity[0] = -velocity[0]
        if position[0] <= self.xMin:
#            position[0] = self.xMin
            position[0] = 2 * self.xMin - position[0]
            velocity[0] = -velocity[0]
        if position[1] >= self.yMax:
#            position[1] = self.yMax
            position[1] = 2 * self.yMax - position[1]
            velocity[1] = -velocity[1]
        if position[1] <= self.yMin:
#            position[1] = self.yMin
            position[1] = 2 * self.yMin - position[1]
            velocity[1] = -velocity[1]

        toWallDistance = np.concatenate([position[0] - self.xBoundary, position[1] - self.yBoundary])
        return position, velocity, toWallDistance    


