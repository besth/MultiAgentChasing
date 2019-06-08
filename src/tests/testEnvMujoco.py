import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node

# Local import
from envMujoco import Reset


# Currently tests only the reset function. Hence, the setup and other methods contain only those parameters that are
# relevant to reset
@ddt
class TestEnvMujoco(unittest.TestCase):
    def setUp(self):
        self.modelName = 'twoAgents'
        self.numAgent = 2

    @data(([0, 0, 0, 0], [0, 0, 0, 0], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))
    @unpack
    def testReset(self, qPosInit, qVelInit, groundTruthReturnedInitialState):
        reset = Reset(self.modelName, qPosInit, qVelInit, self.numAgent)
        returnedInitialState = reset()
        self.assertEqual(returnedInitialState, groundTruthReturnedInitialState)





