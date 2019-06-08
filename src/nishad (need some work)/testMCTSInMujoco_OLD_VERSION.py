# def main():
#     numTrials = 50
#     list_num_simulations = [50, 250, 500, 1000, 1500, 2000]
#
#     envModelName = 'twoAgents'
#     actionSpace = [(10, 0), (-10, 0), (0, 10), (0, -10)]
#     numActionSpace = len(actionSpace)
#     renderOn = False
#     minXDis = 0.2
#     numSimulationFrames = 20
#     isTerminal = IsTerminal(minXDis)
#     transitionFunction = TransitionFunctionNaivePredator(envModelName, isTerminal, renderOn, numSimulationFrames)
#     aliveBouns = 0.05
#     deathPenalty = -1
#     rewardFunction = reward.RewardFunctionCompete(aliveBouns, deathPenalty, isTerminal)
#     reset = Reset(envModelName)
#
#     c_init = 1
#     c_base = 100
#     rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
#     maxRollOutSteps = 10
#
#     calculateScore = CalculateScore(c_init, c_base)
#     selectChild = SelectChild(calculateScore)
#     getActionPrior = GetActionPrior(actionSpace)
#     initializeChildren = InitializeChildren(actionSpace, transitionFunction, getActionPrior)
#     expand = Expand(transitionFunction, isTerminal, initializeChildren)
#     nullRolloutHeuristic = NullRolloutHeuristic()
#     weight = 0.1
#     distanceHeuristicRollout = RolloutHeuristicBasedOnClosenessToTarget(weight)
#     selectNextRoot = SelectNextRoot(transitionFunction)
#
#     # Find the optimal next position of the prey
#     rootAction = (0, 0)
#     initState = reset(2)
#     print("init state: ", initState)
#     rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
#
#     optimalAction = (-10, 0)
#     optimalNextState = transitionFunction(initState, optimalAction)
#     print("optimal next state: ", )
#     optimalPositionOfPreyAfterOneStep = optimalNextState[0][2:4]
#
#     # # Test position of prey after one step of MCTS (vary: number of simulations. Fixed: distance 16, No heuristic)
#     # meanL2Norms = {}
#     # for num_simulations in list_num_simulations:
#     #     print("Number of Simulations: ", num_simulations)
#     #     rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction, rewardFunction, isTerminal, num_simulations, nullRolloutHeuristic)
#     #     mcts = MCTS(num_simulations, selectChild, expand, rollout, backup, selectNextRoot)
#     #     meanL2Norm = DistanceBetweenActualAndOptimalNextPosition(mcts, rootNode, numTrials, optimalPositionOfPreyAfterOneStep)
#     #     meanL2Norms[num_simulations] = meanL2Norm
#     #     print("-------------")
#
#     # # Test position of prey after one step of MCTS (vary: number of simulations. Fixed: distance 16, with heuristic)
#     # meanL2Norms = {}
#     # for num_simulations in list_num_simulations:
#     #     print("Number of Simulations: ", num_simulations)
#     #     rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction, rewardFunction, isTerminal, num_simulations, distanceHeuristicRollout)
#     #     mcts = MCTS(num_simulations, selectChild, expand, rollout, backup, selectNextRoot)
#     #     meanL2Norm = DistanceBetweenActualAndOptimalNextPosition(mcts, rootNode, numTrials, optimalPositionOfPreyAfterOneStep)
#     #     meanL2Norms[num_simulations] = meanL2Norm
#     #     print("-------------")
#
#     # # Test position of prey after one step of MCTS (vary: initial distance between the 2 agents. Fixed: No heuristic. Number of simulations: 100)
#     # num_simulations = 100
#     # listInitialPositions = [[-8, 0, 8, 0], [-6, 0, 6, 0], [-4, 0, 4, 0], [-2, 0, 2, 0]]
#     # meanL2Norms = {}
#     # for initialPositions in listInitialPositions:
#     #     rootAction = (0, 0)
#     #     initState = reset(initialPositions, initVelocities)
#     #     rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
#     #
#     #     optimalAction = (-10, 0)
#     #     optimalNextState = transitionFunction(initState, optimalAction)
#     #     optimalPositionOfPreyAfterOneStep = optimalNextState[0][2:4]
#     #
#     #     print("Initial Positions: ", initPositions)
#     #     print("Optimal Position of prey after one step: ", optimalPositionOfPreyAfterOneStep)
#     #     rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction, rewardFunction, isTerminal, num_simulations, nullRolloutHeuristic)
#     #     mcts = MCTS(num_simulations, selectChild, expand, rollout, backup, selectNextRoot)
#     #     meanL2Norm = DistanceBetweenActualAndOptimalNextPosition(mcts, rootNode, numTrials, optimalPositionOfPreyAfterOneStep)
#     #     meanL2Norms[initialPositions[2]] = meanL2Norm
#     #     print("-------------")
#
#     # Test position of prey after one step of MCTS (vary: initial distance between the 2 agents. Fixed: No heuristic. Number of simulations: 100)
#     num_simulations = 100
#     listInitialPositions = [[-8, 0, 8, 0], [-6, 0, 6, 0], [-4, 0, 4, 0], [-2, 0, 2, 0]]
#     meanL2Norms = {}
#     for initialPositions in listInitialPositions:
#         rootAction = (0, 0)
#         initState = reset(initialPositions, initVelocities)
#         rootNode = Node(id={rootAction: initState}, num_visited=0, sum_value=0, is_expanded=True)
#
#         optimalAction = (-10, 0)
#         optimalNextState = transitionFunction(initState, optimalAction)
#         optimalPositionOfPreyAfterOneStep = optimalNextState[0][2:4]
#
#         print("Initial Positions: ", initPositions)
#         print("Optimal Position of prey after one step: ", optimalPositionOfPreyAfterOneStep)
#         rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunction, rewardFunction, isTerminal, num_simulations, distanceHeuristicRollout)
#         mcts = MCTS(num_simulations, selectChild, expand, rollout, backup, selectNextRoot)
#         meanL2Norm = DistanceBetweenActualAndOptimalNextPosition(mcts, rootNode, numTrials, optimalPositionOfPreyAfterOneStep)
#         meanL2Norms[initialPositions[2]] = meanL2Norm
#         print("-------------")
#
#
#     print("Mean L2 Norms: ", meanL2Norms)
#
#
# if __name__ == "__main__":
#     main()

# (-8, 8) (+8, 8) no heuristic -- {50: 0.2863502884254449, 250: 0.29028531726798934, 500: 0.2411553749530779, 1000: 0.25459040379562237, 1500: 0.2668728893743502, 2000: 0.24230791821689454}
# (-8, 0) (+8, 8) with heuristic -- {50: 0.4750000000000014, 250: 0.47221751442127363, 500: 0.4638700576850902, 1000: 0.47221751442127363, 1500: 0.4750000000000014, 2000: 0.4750000000000014}
# no heuristic: {8: 0.30372034611053356, 6: 0.32387288937435016, 4: 0.24163277400417246, 2: 0.30395912718992385}
# with heuristic: {8: 0.4750000000000014, 6: 0.4750000000000014, 4: 0.4750000000000014, 2: 0.0}
