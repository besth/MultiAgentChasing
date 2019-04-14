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


def makeTitle(xlabel, ylabel):
	return '%s vs %s' % (xlabel, ylabel)


def draw(data):
	plt.figure(figsize=(12, 12))
	usedColor =[]
	neuronList = []
	layerList = []
	colorPair = {}
	numOfFigures = len(list(data.values())[0])
	for keys in data.keys():
		if keys[0] not in neuronList:
			neuronList.append(keys[0])
		if keys[1] not in layerList:
			layerList.append(keys[1])
	for i in range(0, numOfFigures):
		plt.subplot(numOfFigures * 100 + 10 + i + 1)
		for numOfNeuron in neuronList:
			colorPair[numOfNeuron] = randomColor(usedColor)
			usedColor.append(colorPair[numOfNeuron])
			plt.plot([key[1] if key[0]==numOfNeuron else None for key in data.keys()],
						[data[key][i] if key[0]==numOfNeuron else None for key in data.keys()],
						label=str(numOfNeuron), color=colorPair[numOfNeuron])
			plt.xlabel('Layer')
			plt.ylabel('Value%s' % i)
			plt.legend(loc='upper left')
			# plt.xticks(np.arange(0, max(layerList), step=1))
			title = makeTitle('Value%s'%i, 'Layer')
			plt.title(title)
			plt.savefig(title)


if __name__ == '__main__':
	data = {(200,2): [10, -20], (200,4): [20,30], (200,8): [30,40], (200,16):[40,50],
			(400,2): [15,25], (400,4): [25,35], (400,8): [35,45], (400,16): [45,55],
			(800,2): [18,28], (800,4): [28,38], (800,8): [38,48], (800,16): [48,58],
			(1000,2): [22,32], (1000,4): [32,42], (1000,8): [42,52], (1000,16): [52,62]}
	draw(data)