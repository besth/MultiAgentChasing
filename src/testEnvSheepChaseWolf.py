import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack

# Local import
from envSheepChaseWolf import computeDistance, DistanceBetweenActualAndOptimalNextPosition


@ddt
class TestDistanceBetweenOptimalAndActualNextPosition(unittest.TestCase):
    @data(([0, 0], [1, 1], np.sqrt(2)), ([0, 0], [1, 0], 1), ([-10, -10], [10, 10], 20*np.sqrt(2)), ([3, -4], [-4, 3], 7*np.sqrt(2)))
    @unpack
    def testComputeDistance(self, pos1, pos2, groundTruthDistance):
        distance = computeDistance(np.asarray(pos1), np.asarray(pos2))
        self.assertAlmostEqual(groundTruthDistance, distance)

    @data(
        (np.asarray([2, 0]), [[(np.asarray([[1, 0, 1, 0, 0, 0], [3, 0, 3, 0, 0, 0]]), np.asarray([[1, 0], [0, 0]])),
                               (np.asarray([[2, 0, 2, 0, 0, 0], [3, 0, 3, 0, 0, 0]]), np.asarray([[1, 0], [0, 0]])),
                               (np.asarray([[3, 0, 3, 0, 0, 0], [3, 0, 3, 0, 0, 0]]), np.asarray([[1, 0], [0, 0]]))]],
         0, 0),
        (np.asarray([-9, -10]), [[(np.asarray([[-10, -10, -10, -10, 0, 0], [-8, -10, -8, -10, 0, 0]]), np.asarray([[1, 0], [0, 0]])),
                               (np.asarray([[-9, -10, -9, -10, 0, 0], [-8, -10, -8, -10, 0, 0]]), np.asarray([[1, 0], [0, 0]])),
                               (np.asarray([[-8, -10, -8, -10, 0, 0], [-8, -10, -8, -10, 0, 0]]), np.asarray([[1, 0], [0, 0]]))],
                                 [(np.asarray([[-10, -10, -10, -10, 0, 0], [-8, -10, -8, -10, 0, 0]]), np.asarray([[0, 1], [0, 0]])),
                               (np.asarray([[-10, -9, -10, -9, 0, 0], [-8, -10, -8, -10, 0, 0]]), np.asarray([[1, 0], [0, 0]])),
                               (np.asarray([[-9, -9, -9, -9, 0, 0], [-8, -10, -8, -10, 0, 0]]), np.asarray([[1, 0], [0, 0]]))]],
         1/np.sqrt(2), 1/np.sqrt(2)),
        )
    @unpack
    def testDistanceBetweenOptimalAndActualNextPosition(self, optimalNextPosition, trajectories, groundTruthMeanDistance, groundTruthStdDev):
        nextPositionIndex = 1
        stateIndexInTuple = 0
        xPosIndex = 2
        numXPosEachAgent = 2
        agentID = 0

        measurementFunction = DistanceBetweenActualAndOptimalNextPosition(optimalNextPosition, nextPositionIndex,
                                                                          stateIndexInTuple, xPosIndex,
                                                                          numXPosEachAgent, agentID)

        [meanDistance, distanceStdDev] = measurementFunction(trajectories)

        self.assertAlmostEqual(meanDistance, groundTruthMeanDistance)
        self.assertAlmostEqual(distanceStdDev, groundTruthStdDev)



