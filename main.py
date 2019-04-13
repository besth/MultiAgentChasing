import tensorflow as tf
import numpy as np
import functools as ft
import env
import reward
import tensorflow_probability as tfp
import random
import agentsEnv as ag
import itertools as it
import pygame as pg
import offlineA2CMonteCarloAdvantageDiscrete as A2CMC
from pydoc import locate


def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)
    
    actionSpace = [[10,0],[7,7],[0,10],[-7,7],[-10,0],[-7,-7],[0,-10],[7,-7]]
    numActionSpace = len(actionSpace)
    numStateSpace = 4

    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([119, 119])
    initWolfPositionNoise = np.array([59, 59])
    sheepPositionAndVelocityReset = ag.SheepPositionAndVelocityReset(initSheepPosition, initSheepVelocity, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionAndVelocityReset = ag.WolfPositionAndVelocityReset(initWolfPosition, initWolfVelocity, initWolfPositionNoise, checkBoundaryAndAdjust)
    
    numOneAgentState = 2
    positionIndex = [0, 1]
    velocityIndex = [2, 3]
    sheepVelocitySpeed = 10
    sheepActionFrequency = 1
    wolfVelocitySpeed = 6
    wolfActionFrequency = 12
    sheepPositionAndVelocityTransation = ag.SheepPositionAndVelocityTransation(sheepVelocitySpeed, sheepActionFrequency, 
            numOneAgentState, positionIndex, velocityIndex, checkBoundaryAndAdjust) 
    wolfPositionAndVelocityTransation = ag.WolfPositionAndVelocityTransation(wolfVelocitySpeed, wolfActionFrequency,
            numOneAgentState, positionIndex, velocityIndex, checkBoundaryAndAdjust) 
    
    numAgent = 2
    sheepIndexOfId = 0
    wolfIndexOfId = 1
    originAgentId = list(range(numAgent))
    #fixedId for sheep
    fixedIds= list(range(0, 1))
    #unfixedId for wolf and distractors
    unfixedIds = list(range(1, numAgent))
    possibleUnfixedIds = it.permutations(unfixedIds)
    possibleAgentIds = [fixedIds + list(unfixedIds) for unfixedIds in possibleUnfixedIds]
    possibleWolfSubtleties = [50]
    conditions = it.product(possibleAgentIds, possibleWolfSubtleties)
    transitionFunctions = [env.TransitionFunction(agentIds, sheepIndexOfId, wolfIndexOfId, wolfSubtlety, 
        sheepPositionAndVelocityReset, wolfPositionAndVelocityReset, sheepPositionAndVelocityTransation, wolfPositionAndVelocityTransation) 
        for agentIds, wolfSubtlety in conditions]
    
    minDistance = 15
    isTerminals = [env.IsTerminal(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex, 
        minDistance) for agentIds in possibleAgentIds]
     
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255,255,255]
    circleColorList = [[50,255,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50]]
    circleSize = 8
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)

    aliveBouns = -1
    deathPenalty = 20
    rewardDecay = 0.99
    rewardFunctions = [reward.RewardFunctionTerminalPenalty(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex,
        aliveBouns, deathPenalty, isTerminal) for agentIds, isTerminal in zip(possibleAgentIds, isTerminals)] 
    accumulateReward = AccumulateReward(rewardDecay)
    
    maxTimeStep = 150
    sampleTrajectories = [A2CMC.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal) for transitionFunction, isTerminal in zip(transitionFunctions, isTerminals)]

    approximatePolicy = A2CMC.ApproximatePolicy(actionSpace)
    trainCritic = A2CMC.TrainCriticMonteCarloTensorflow(accumulateReward) 
    #trainCritic = TrainCriticBootstrapTensorflow(rewardDecay) 
    estimateAdvantage = A2CMC.EstimateAdvantageMonteCarlo(accumulateReward) 
    trainActor = A2CMC.TrainActorMonteCarloTensorflow(actionSpace) 
    
    numTrajectory = 100
    maxEpisode = 100000
    modelTrain = OfflineAdvantageActorCritic(numTrajectory, maxEpisode, render)

    # Load algorithm class
    algorithm_name = "PolicyGradient"
    algorithm_class = locate("algorithms.{}.{}".format(algorithm_name, algorithm_name))
    algorithm = algorithm_class((numAgent,
                                 maxEpisode, 
                                 maxTimeStep,  
                                 transitionFunction, 
                                 isTerminal, 
                                 reset,  
                                 saveRate))

    trained_models = algorithm(models)

    # Evaluate models 

    print("Success.")

if __name__ == "__main__":
    main()
