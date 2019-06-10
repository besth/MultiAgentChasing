import sys
sys.path.append('algorithms/')
import unittest
import numpy as np
from ddt import ddt, data, unpack

# Local import
from mcts import RolloutHeuristicBasedOnClosenessToTarget
from testMCTSPriorEffectInMujoco import DirectionalPrior


@ddt
class TestMCTSInMujoco(unittest.TestCase):
    @data((np.asarray([[-8, 0, -8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), -1.6), (np.asarray([[8, 0, 8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), 0), (np.asarray([[10, -10, 10, -10, 0, 0], [-10, 10, -10, 10, 0, 0]]), -2*np.sqrt(2)))
    @unpack
    def testRolloutHeuristicBasedOnClosenessToTarget(self, state, groundTruthReward):
        weight = 0.1
        sheepId = 0
        wolfId = 1
        xPosIndex = 2
        numXPosEachAgent = 2
        rolloutHeuristic = RolloutHeuristicBasedOnClosenessToTarget(weight, sheepId, wolfId, xPosIndex, numXPosEachAgent)
        reward = rolloutHeuristic(state)
        self.assertEqual(reward, groundTruthReward)


    @data((np.asarray([[-8, 0, -8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), {(10, 0): 0.9, (7, 7): 0.1/7, (0, 10): 0.1/7, (-7, 7): 0.1/7, (-10, 0): 0.1/7, (-7, -7): 0.1/7, (0, -10): 0.1/7, (7, -7): 0.1/7}),
          (np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
           {(10, 0): 1/8, (7, 7): 1/8, (0, 10): 1/8, (-7, 7): 1/8, (-10, 0): 1/8, (-7, -7): 1/8,
            (0, -10): 1/8, (7, -7): 1/8}))
    @unpack
    def testNonUniformPriorFunction(self, currentState, groundTruthPrior):
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        sheepId = 0
        wolfId = 1
        qPosIndex = 0
        numQPosEachAgent = 2
        priorForMostPreferredAction = 0.9
        sheepInitQPos = [-8, 0]
        mostPrefferedAction = (10, 0)

        nonUniformPrior = DirectionalPrior(actionSpace, sheepInitQPos, mostPrefferedAction, sheepId, wolfId, qPosIndex, numQPosEachAgent, priorForMostPreferredAction)
        prior = nonUniformPrior(currentState)
        orederedListPriorProbabilities = [prior[action] for action in actionSpace]
        groundTruthOrederedListPriorProbabilities = [groundTruthPrior[action] for action in actionSpace]

        np.testing.assert_array_almost_equal(orederedListPriorProbabilities, groundTruthOrederedListPriorProbabilities)


