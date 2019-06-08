import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node
from matplotlib import pyplot as plt

# Local import
from algorithms.mcts import CalculateScore, SelectChild, Expand, RollOut, backup, GetActionPrior, MCTS, InitializeChildren, SelectNextRoot
from envMujoco import Reset, TransitionFunctionNaivePredator, IsTerminal
import reward

@ddt
class TestMCTSInMujoco(unittest.TestCase):
    def setUp(self):
        self.numTrials = 50
        self.num_simulations = 250

        self.envModelName = 'twoAgents'
        self.actionSpace = [(10, 0), (-10, 0), (0, 10), (0, -10), (7, 7), (7, -7), (-7, 7), (-7, -7)]
        self.numActionSpace = len(self.actionSpace)
        self.renderOn = False
        self.minXDis = 0.2
        self.numSimulationFrames = 20
        self.isTerminal = IsTerminal(self.minXDis)
        self.transitionFunction = TransitionFunctionNaivePredator(self.envModelName, self.isTerminal, self.renderOn, self.numSimulationFrames)
        self.aliveBouns = 0.05
        self.deathPenalty = -1
        self.rewardFunction = reward.RewardFunctionCompete(self.aliveBouns, self.deathPenalty, self.isTerminal)
        self.reset = Reset(self.envModelName, 2)

        self.c_init = 1
        self.c_base = 100
        self.rolloutPolicy = lambda state: self.actionSpace[np.random.choice(range(self.numActionSpace))]
        self.maxRollOutSteps = 10
        self.rollout_heuristic = RollOut(self.rolloutPolicy, self.maxRollOutSteps, self.transitionFunction, self.rewardFunction, self.isTerminal,
                          self.num_simulations, True)
        self.rollout_no_heuristic = RollOut(self.rolloutPolicy, self.maxRollOutSteps, self.transitionFunction,
                                         self.rewardFunction, self.isTerminal,
                                         self.num_simulations, False)

        self.calculateScore = CalculateScore(self.c_init, self.c_base)
        self.selectChild = SelectChild(self.calculateScore)
        self.getActionPrior = GetActionPrior(self.actionSpace)
        self.initializeChildren = InitializeChildren(self.actionSpace, self.transitionFunction, self.getActionPrior)
        self.expand = Expand(self.isTerminal, self.initializeChildren)
        self.selectNextRoot = SelectNextRoot(self.transitionFunction)


    def DistanceBetweenActualAndOptimalNextPosition(self, mcts, rootNode, numTrials, optimalNextPositionOfPrey):  # rename
        L2NormsEachTrial = []

        for trial in range(numTrials):
            nextRoot = mcts(rootNode)
            nextPositionOfPrey = list(nextRoot.id.values())[0][0][2:4]

            # L2 norm between actual and optimal next step
            L2Norm = np.linalg.norm((optimalNextPositionOfPrey - nextPositionOfPrey), ord=2)

            L2NormsEachTrial.append(L2Norm)

        meanL2Norm = np.mean(np.array(L2NormsEachTrial))

        return meanL2Norm


    # This tests sheep chasing wolf and not the other way round
    def testNumSimulationEffectsOnFirstStep(self):
        list_num_simulations = [0, 50, 100, 200, 500]

        # reverse the rewards because we want the sheep to chase the wolf in this case
        self.aliveBouns = -0.05
        self.deathPenalty = 1

        # Find the optimal next position of the prey
        rootAction = (0, 0)
        initState = self.reset(2)
        print("Initial State: ", initState)
        rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
        optimalAction = (10, 0)
        optimalNextState = self.transitionFunction(initState, optimalAction)
        optimalPositionOfPreyAfterOneStep = optimalNextState[0][2:4]
        print("Optimal Next State: ", optimalNextState)

        meanL2Norms_NoHeuristic = {}
        meanL2Norms_Heuristic = {}
        for num_simulations in list_num_simulations:
            print("Number of Simulations: ", num_simulations)
            mcts_heuristic = MCTS(num_simulations, self.selectChild, self.expand, self.rollout_heuristic, backup, self.selectNextRoot)
            mcts_no_heuristic = MCTS(num_simulations, self.selectChild, self.expand, self.rollout_no_heuristic, backup,
                                  self.selectNextRoot)
            meanL2Norm_Heuristic = self.DistanceBetweenActualAndOptimalNextPosition(mcts_heuristic, rootNode, self.numTrials, optimalPositionOfPreyAfterOneStep)
            meanL2Norm_NoHeuristic = self.DistanceBetweenActualAndOptimalNextPosition(mcts_no_heuristic, rootNode,
                                                                                    self.numTrials,
                                                                                    optimalPositionOfPreyAfterOneStep)
            meanL2Norms_Heuristic[num_simulations] = meanL2Norm_Heuristic
            meanL2Norms_NoHeuristic[num_simulations] = meanL2Norm_NoHeuristic
            print("-------------")

        # print the results
        print("Mean L2 Norms (No Heuristic): ", meanL2Norms_NoHeuristic)
        print("Mean L2 Norms (With Heuristic): ", meanL2Norms_Heuristic)


        # plot the results
        coordinatePairsList = sorted(meanL2Norms_NoHeuristic.items())
        x, y = zip(*coordinatePairsList)
        plt.plot(x, y, label='Without Heuristic')

        coordinatePairsList = sorted(meanL2Norms_Heuristic.items())
        x, y = zip(*coordinatePairsList)
        plt.plot(x, y, label='With Heuristic')

        plt.xlabel("Number of Simulations")
        plt.ylabel("Mean L2 Norm from optimal next position")
        plt.title("Mean L2 Norm from optimal next position vs. Number of Simulations")

        plt.legend(loc='upper left')
        plt.show()

























# These have the wrong optimal action so they are practically wrong
# Mean L2 Norms (No Heuristic):  {250: 0.18087461126426813, 750: 0.18087461126426813, 1250: 0.31226695321755316, 1750: 0.3970162643392009}
# Mean L2 Norms (With Heuristic):  {250: 0.18087461126426813, 750: 0.18087461126426813, 1250: 0.26674057048332583, 1750: 0.3649320575509297}

# These have the right optimal action
# Mean L2 Norms (No Heuristic):  {0: 0.3537467064040039, 50: 0.3190155648254278, 100: 0.33244161987637055, 200: 0.3490721507251207, 500: 0.2962804693949228}
# Mean L2 Norms (With Heuristic):  {0: 0.2834805137868965, 50: 0.29503583909148223, 100: 0.31610989502047365, 200: 0.33103931538950626, 500: 0.3241398151393979}


