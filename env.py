import os
import numpy as np
import pandas as pd
import GenerateModel as gm
import pygame as pg
import itertools as it
import random 
#np.random.seed(123)

class TransitionFunction():
    def __init__(self, agentIds, sheepIndexOfId, wolfIndexOfId, wolfSubtlety, sheepPositionAndVelocityReset, wolfPositionAndVelocityReset, sheepPositionAndVelocityTransation, wolfPositionAndVelocityTransation):
        self.agentIds = agentIds
        self.sheepId = self.agentIds[sheepIndexOfId]
        self.wolfId = self.agentIds[wolfIndexOfId] 
        self.wolfSubtlety = wolfSubtlety 
        self.sheepPositionAndVelocityReset = sheepPositionAndVelocityReset
        self.wolfPositionAndVelocityReset = wolfPositionAndVelocityReset
        self.sheepPositionAndVelocityTransation = sheepPositionAndVelocityTransation
        self.wolfPositionAndVelocityTransation = wolfPositionAndVelocityTransation

    def __call__(self, oldState, action):
        if oldState is None:
            sheepPositionAndVelocity = self.sheepPositionAndVelocityReset()
            wolfPositionAndVelocity = self.wolfPositionAndVelocityReset()
            self.timeStep = -1 
        else:
            sheepPositionAndVelocity = self.sheepPositionAndVelocityTransation(oldState, self.sheepId, action, self.timeStep) 
            wolfPositionAndVelocity = self.wolfPositionAndVelocityTransation(oldState, self.sheepId, self.wolfId, self.wolfSubtlety, self.timeStep)
        fixedOrderNewState = np.array([sheepPositionAndVelocity, wolfPositionAndVelocity])
        orderFollowIdNewState = fixedOrderNewState[self.agentIds]
        newState = np.concatenate(orderFollowIdNewState)
        self.timeStep = self.timeStep + 1
        return newState

class IsTerminal():
    def __init__(self, agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex, minDistance):
        self.agentIds = agentIds
        self.sheepId = self.agentIds[sheepIndexOfId]
        self.wolfId = self.agentIds[wolfIndexOfId] 
        self.numOneAgentState = numOneAgentState 
        self.positionIndex = positionIndex
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        sheepState = state[self.numOneAgentState * self.sheepId : self.numOneAgentState * (self.sheepId + 1)]
        sheepPosition = sheepState[min(self.positionIndex) : max(self.positionIndex) + 1]
        wolfState = state[self.numOneAgentState * self.wolfId : self.numOneAgentState * (self.wolfId + 1)]
        wolfPosition = wolfState[min(self.positionIndex) : max(self.positionIndex) + 1]
        if np.sum(np.power(sheepPosition - wolfPosition, 2)) ** 0.5 <= self.minDistance:
            terminal = True
        return terminal   

class Render():
    def __init__(self, numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize,saveImage,saveImageFile):
        self.numAgent = numAgent
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageFile = saveImageFile
    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
            self.screen.fill(self.screenColor)
            for i in range(self.numAgent):
                oneAgentState = state[self.numOneAgentState * i : self.numOneAgentState * (i + 1)]
                oneAgentPosition = oneAgentState[min(self.positionIndex) : max(self.positionIndex) + 1]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
            pg.display.flip()
            currentDir = os.getcwd()
            parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
            saveImageDir=parentDir+'/data/'+self.saveImageFile
            if self.saveImage==True:
                filenameList = os.listdir(saveImageDir)
                pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
            pg.time.wait(17)


