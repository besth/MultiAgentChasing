import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def makeTitle(xlabel, ylabel, graphIndex):
	return '%s%s vs %s' % (xlabel, graphIndex, ylabel)


def splitDictionary(originalDict, splitFactor):
	newDict = [{key: value[index] for key, value in originalDict.items()} for index in range(splitFactor)]
	return newDict


def dictToDataframe(data, axisName, lineVariableIndex):
	numOfDependentVariable = len(list(data.values())[0])
	splitedDependentVaraibles = splitDictionary(data, numOfDependentVariable)
	dataDFs = [pd.Series(dictionary).rename_axis(axisName).unstack(level=lineVariableIndex) for dictionary in splitedDependentVaraibles]
	return dataDFs


def drawPerGraph(dataDF, title):
	plt.title(title)
	dataDF.plot(title=title)
	plt.savefig(title)


def draw(data, independetVariablesName, lineVariableIndex=0, xVariableIndex=1):
	dataDFs = dictToDataframe(data, independetVariablesName, lineVariableIndex)
	plt.figure()
	titles = [makeTitle('Value', independetVariablesName[xVariableIndex], graphIndex=index) for index in range(len(dataDFs))]
	[drawPerGraph(dataDF, title) for dataDF, title in zip(dataDFs, titles)]


if __name__ == '__main__':
	num_sim = [100, 200, 500, 1000, 1500, 2000]

	rollout_reach_target_prob_no_heuristic_distance = [0.954047619047619, 0.9104166666666667, 0.8246078431372549, 0.6477848101265823, 0.46820754716981133, 0.41257692307692306]
	distance = [1, 2, 4, 8, 12, 16]

	rollout_reach_target_prob_no_heuristic_simulation_distance16 = [0.3172, 0.4126, 0.5807, 0.6829]
	rollout_reach_target_prob_with_heuristic_distance = [0.9712727272727273, 0.9503888888888888, 0.9300851063829787, 0.8400821917808219, 0.7460222222222223, 0.6762222222222222]

	mean_first_step_distance_to_optimal_no_heuristic = [0.1751776429621799, 0.15343741661206808, 0.10593741661206812, 0.1198498445057071, 0.07234984450570713, 0.01808746112642678]
	mean_first_step_distance_to_optimal_with_heuristic_test50 = [0.1520751831034078, 0.08320232118156319, 0.08681981340684855, 0.08991983560283538, 0.05064489115399499, 0.05787987560456569]
	mean_first_step_distance_to_optimal_no_heuristic_test50 = [0.20149913343136788, 0.20104943879628082, 0.140337871969986, 0.11110731892524901, 0.09631981340684856, 0.08108261325336098]

	plt.figure()
	# plt.plot(distance, prob_heuristic)

	plt.plot(num_sim, mean_first_step_distance_to_optimal_no_heuristic_test50, label="no heuristic")
	plt.plot(num_sim, mean_first_step_distance_to_optimal_with_heuristic_test50, label="with heuristic")
	plt.legend()
	plt.title("Effect of rollout heuristic on first step optimality")
	# plt.xlabel("Distances")
	plt.xlabel("Number of simulations")
	# plt.ylabel("Percentage of rollout reached target")
	plt.ylabel("Mean distance (Max possible: 0.43; Min possible: 0.00)")
	plt.savefig("first_step_distance_comparison.png")
	# plt.show()
