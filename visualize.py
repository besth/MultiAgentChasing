import matplotlib.pyplot as plt
import random
import numpy as np


def randomColor(usedColor):
	colorSet = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
	color = ""
	for i in range(6):
		color += colorSet[random.randint(0, 14)]
	while color in usedColor:
		for i in range(6):
			color += colorSet[random.randint(0,14)]
	return '#'+color


def makeTitle():
	return 'Value vs Layer'


def draw(data):
	plt.figure(figsize=(12, 12))
	usedColor =[]
	maxValue = 0
	maxLayer = 0
	for numOfNeuron in data.keys():
		color = randomColor(usedColor)
		usedColor.append(color)
		numOfLayerDict = data[numOfNeuron]
		if maxValue < max(numOfLayerDict.values()):
			maxValue = max(numOfLayerDict.values())
		if maxLayer < max(numOfLayerDict.keys()):
			maxLayer = max(numOfLayerDict.keys())
		# print([numOfLayerDict[layer] for layer in numOfLayerDict.keys()])
		plt.plot([layer for layer in numOfLayerDict.keys()], [numOfLayerDict[layer] for layer in numOfLayerDict.keys()], label=str(numOfNeuron), color=color)
	plt.xlabel('layer')
	plt.ylabel('value')
	plt.legend(loc='upper left')
	plt.xticks(np.arange(0, maxLayer, step=1))
	plt.yticks(np.arange(0, maxValue, step=1))
	title = makeTitle()
	plt.title(title)
	plt.show()
	plt.savefig(title)


if __name__ == '__main__':
	data = {200: {2: 10, 4: 20, 8: 30, 16: 40},
			400: {2: 15, 4: 25, 8: 35, 16: 45},
			800: {2: 18, 4: 28, 8: 38, 16: 48},
			1000: {2: 22, 4: 32, 8: 42, 16: 52},
			}
	draw(data)