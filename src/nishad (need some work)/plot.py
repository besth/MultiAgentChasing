from matplotlib import pyplot as plt

meanL2Norms_NoHeuristic = {0: 0.3537467064040039, 50: 0.3190155648254278, 100: 0.33244161987637055, 200: 0.3490721507251207, 500: 0.2962804693949228}
meanL2Norms_Heuristic = {0: 0.2834805137868965, 50: 0.29503583909148223, 100: 0.31610989502047365, 200: 0.33103931538950626, 500: 0.3241398151393979}

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



