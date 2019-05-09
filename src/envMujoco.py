import mujoco_py as mujoco
import os
import numpy as np

class Reset():
    def __init__(self, modelName, qPosInitNoise=0, qVelInitNoise=5):
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
    def __call__(self, numAgent):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        numQPosEachAgent = int(numQPos/numAgent)
        numQVelEachAgent = int(numQVel/numAgent)

        qPos = self.simulation.data.qpos + np.random.uniform(low = -self.qPosInitNoise, high = self.qPosInitNoise, size = numQPos)
        qVel = self.simulation.data.qvel + np.random.uniform(low = -self.qVelInitNoise, high = self.qVelInitNoise, size = numQVel)
        qVel = [qVel[0], qVel[1], 0, 0]
        # print(qVel)
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()
        xPos = np.concatenate(self.simulation.data.body_xpos[-numAgent: , :numQPosEachAgent])
        startState = np.array([np.concatenate([qPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], xPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
            qVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(numAgent)]) 
        return startState

class TransitionFunctionNaivePredator():
    def __init__(self, modelName, isTerminal, renderOn): 
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
        self.renderOn = renderOn
        if self.renderOn:
            self.viewer = mujoco.MjViewer(self.simulation)
            self.frames = []

        self.isTerminal = isTerminal

    def __call__(self, allAgentOldState, preyAction, renderOpen = False, numSimulationFrames = 20):
        numAgent = len(allAgentOldState)
        numQPosEachAgent = int(self.numQPos/numAgent)
        numQVelEachAgent = int(self.numQVel/numAgent)

        preyState = allAgentOldState[0][numQPosEachAgent: numQPosEachAgent + 2]
        predatorState = allAgentOldState[1][numQPosEachAgent: numQPosEachAgent + 2]

        # predatorAction = preyState - predatorState
        predatorAction = (0, 0)
        # print("predator action", predatorAction)
        predatorActionNorm = np.sum(np.abs(predatorAction))
        if predatorActionNorm != 0:
            predatorAction /= predatorActionNorm

        # predatorAction *= 5

        # print("normalized predator action", predatorAction)
        # print("prey action", preyAction)

        # preyActionNorm = np.sum(np.abs(preyAction))
        # if preyActionNorm != 0:
        #     preyAction /= preyActionNorm

        # print("normalized prey action", preyAction)

        allAgentAction = np.array(preyAction)
        allAgentAction = np.append(allAgentAction, predatorAction)


        allAgentOldQPos = allAgentOldState[:, 0:numQPosEachAgent].flatten()
        allAgentOldQVel = allAgentOldState[:, -numQVelEachAgent:].flatten()

        # print(allAgentAction)

        # Hacking
        # allAgentOldQVel[2] = predatorAction[0]*10
        # allAgentOldQVel[3] = predatorAction[1]*10
        # allAgentAction[-2:] = [0, 0]

        self.simulation.data.qpos[:] = allAgentOldQPos
        self.simulation.data.qvel[:] = allAgentOldQVel
        self.simulation.data.ctrl[:] = allAgentAction.flatten()

        # print(allAgentAction)
        
        for i in range(numSimulationFrames):
            self.simulation.step()
            if self.renderOn:
                frame = self.simulation.render(1024, 1024, camera_name="center")#, mode="window")
                self.frames.append(frame)
            
            newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
            newXPos = np.concatenate(self.simulation.data.body_xpos[-numAgent: , :numQPosEachAgent])
            newState = np.array([np.concatenate([newQPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], newXPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
                        newQVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(numAgent)]) 
            
            if self.isTerminal(newState):
                break

        return newState


# class TransitionFunction():
#     def __init__(self, modelName, renderOn):
#         model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
#         self.simulation = mujoco.MjSim(model)
#         self.numQPos = len(self.simulation.data.qpos)
#         self.numQVel = len(self.simulation.data.qvel)
#         self.renderOn = renderOn
#         if self.renderOn:
#             self.viewer = mujoco.MjViewer(self.simulation)
#     def __call__(self, allAgentOldState, allAgentAction, renderOpen = False, numSimulationFrames = 100):
#         numAgent = len(allAgentOldState)
#         numQPosEachAgent = int(self.numQPos/numAgent)
#         numQVelEachAgent = int(self.numQVel/numAgent)
#
#         allAgentOldQPos = allAgentOldState[:, 0:numQPosEachAgent].flatten()
#         allAgentOldQVel = allAgentOldState[:, -numQVelEachAgent:].flatten()
#
#         self.simulation.data.qpos[:] = allAgentOldQPos
#         self.simulation.data.qvel[:] = allAgentOldQVel
#         self.simulation.data.ctrl[:] = allAgentAction.flatten()
#
#         frames = list()
#         for i in range(numSimulationFrames):
#             self.simulation.step()
#             if self.renderOn:
#                 # self.viewer.render()
#                 print("into render")
#                 currFrame = self.simulation.render()
#                 frames.append(currFrame)
#
#         if len(frames) != 0:
#             skvideo.io.vwrite("./video.mp4", np.asarray(frames))
#             print("video saved")
#
#         newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
#         newXPos = np.concatenate(self.simulation.data.body_xpos[-numAgent: , :numQPosEachAgent])
#         newState = np.array([np.concatenate([newQPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], newXPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
#             newQVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(numAgent)])
#         return newState

def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class IsTerminal():
    def __init__(self, minXDis):
        self.minXDis = minXDis
    def __call__(self, state):
        # Assume only two agents. get x position
        pos0 = state[0][2:4]
        pos1 = state[1][2:4]
        distance = euclideanDistance(pos0, pos1)
        terminal = (distance <= 2 * self.minXDis)
        return terminal