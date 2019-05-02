import matplotlib.pyplot as plt
import pandas as pd


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
    data = {(1, -1): [1.00], (1, 5): [0.7625], (1, 60): [0.650], (1, 120): [0.725], (1, 180): [1],
            (4, -1): [1.00], (4, 5): [0.8750], (4, 60): [0.875], (4, 120): [0.875], (4, 180): [1],
            (16, -1): [0.975], (16, 5): [0.80], (16, 60): [0.8625], (16, 120): [0.975], (16, 180): [1]}
    #data = {(128, 2): [10, 20], (128, 4): [10, 30]}
    draw(data, ['numTree', 'chasingSubtlety'])
