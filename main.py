
import tensorflow as tf
from pydoc import locate


def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)
    
    numAgent = 2
    numActionSpace = 2
    numStateSpace = 6
    actionLow = np.array([-10, -10])
    actionHigh = np.array([10, 10])
    actionRatio = (actionHigh - actionLow) / 2.
    actionNoise = np.array([100.0, 100.0])
    noiseDecay = 0.999

    # numAgents = 1
    # numOtherActionSpace = (numAgents - 1) * numActionSpace
    
    envModelName = 'twoAgentsChasing'
    renderOn = True
    restore = False
    maxTimeStep = 500
    minXDis = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    catchReward = 50
    disRewardDiscount = 0.2
    rewardDecay = 0.99

    memoryCapacity = 100000
    numMiniBatch = 2500

    maxEpisode = 100000
    saveRate = 100

    # Get models.
    models = None

    transitionFunction = env.TransitionFunctionNaivePredator(envModelName, renderOn)
    isTerminal = env.IsTerminal(minXDis)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)

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
