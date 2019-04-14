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


# Local import
import offlineA2CMonteCarloAdvantageDiscrete as A2CMC
from model import GenerateActorCriticModel
from evaluate import Evaluate
from visualize import draw



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
    initSheepPositionNoise = np.array([150, 150])
    initWolfPositionNoise = np.array([60, 60])
    sheepPositionAndVelocityReset = ag.SheepPositionAndVelocityReset(initSheepPosition, initSheepVelocity, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionAndVelocityReset = ag.WolfPositionAndVelocityReset(initWolfPosition, initWolfVelocity, initWolfPositionNoise, checkBoundaryAndAdjust)
    
    numOneAgentState = 2
    positionIndex = [0, 1]
    velocityIndex = [2, 3]
    sheepVelocitySpeed = 10
    sheepActionFrequency = 1
    wolfVelocitySpeed = 0
    wolfActionFrequency = 1
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
    accumulateReward = A2CMC.AccumulateReward(rewardDecay)
    
    maxTimeStep = 150
    sampleTrajectories = [A2CMC.SampleTrajectory(maxTimeStep, transitionFunction, isTerminal) for transitionFunction, isTerminal in zip(transitionFunctions, isTerminals)]

    approximatePolicy = A2CMC.ApproximatePolicy(actionSpace)
    approximateValue = A2CMC.approximateValue
    trainCritic = A2CMC.TrainCriticMonteCarloTensorflow(accumulateReward) 
    #trainCritic = TrainCriticBootstrapTensorflow(rewardDecay) 
    estimateAdvantage = A2CMC.EstimateAdvantageMonteCarlo(accumulateReward) 
    trainActor = A2CMC.TrainActorMonteCarloTensorflow(actionSpace) 
    
    numTrajectory = 2
    maxEpisode = 2

    # Generate models.
    learningRateActor = 1e-4
    learningRateCritic = 1e-4
    # hiddenNeuronNumbers = [128, 256, 512, 1024]
    # hiddenDepths = [2, 4, 8]
    hiddenNeuronNumbers = [128]
    hiddenDepths = [2,4]
    generateModel = GenerateActorCriticModel(numStateSpace, numActionSpace, learningRateActor, learningRateCritic)
    modelDict = {(n, d): generateModel(d, round(n/d)) for n, d in it.product(hiddenNeuronNumbers, hiddenDepths)}

    print("Generated graphs")
    # Train.
    actorCritic = A2CMC.OfflineAdvantageActorCritic(numTrajectory, maxEpisode, render)
    modelTrain = lambda  actorModel, criticModel: actorCritic(actorModel, criticModel, approximatePolicy, sampleTrajectories, rewardFunctions, trainCritic,
            approximateValue, estimateAdvantage, trainActor)
    trainedModelDict = {key: modelTrain(model[0], model[1]) for key, model in modelDict.items()}

    print("Finished training")
    # Evaluate
    modelEvaluate = Evaluate(numTrajectory, approximatePolicy, sampleTrajectories[0], rewardFunctions[0])
    meanEpisodeRewards = {key: modelEvaluate(model[0], model[1]) for key, model in trainedModelDict.items()}

    print("Finished evaluating")
    # Visualize
    draw(meanEpisodeRewards)

if __name__ == "__main__":
    main()
