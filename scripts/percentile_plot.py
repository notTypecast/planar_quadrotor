from os import mkdir
from os.path import isdir
from sys import argv
from matplotlib import pyplot as plt
import numpy as np

# Make directory to save plots
if not isdir("plots"):
    mkdir("plots")

# Read data from file
with open(argv[1], "r") as f:
    params = next(f).split()
    data = []
    for line in f:
        data.append(list(map(float, line.split())))

    data = np.array(data)

# Parse parameters
g = float(params[0])
steps = int(params[1])
episodes = int(params[2])
runs = int(params[3])
target = (float(params[4]), float(params[5]))

# Average the data for each step of each episode
episode_data = np.zeros((episodes, steps, runs))

for i in range(runs):
    run_idx = i*episodes
    for j in range(episodes):
        for k in range(steps):
            episode_data[j][k][i] = data[run_idx + j][k]

mean_data = np.mean(episode_data, axis=2)
perc1 = np.percentile(episode_data, 25, axis=2)
perc2 = np.percentile(episode_data, 75, axis=2)

steps_arr = np.arange(1, steps+1)

for i in range(episodes):
    if mean_data[i][0] == 0:
        break
    plt.plot(steps_arr, mean_data[i], label=f"Episode {i+1}")
    plt.fill_between(steps_arr, perc1[i], perc2[i], alpha=0.2)

plt.xlabel("Steps")
plt.ylabel("Error")
plt.title(f"g = {g}")
plt.legend()
plt.grid()
#plt.show()
plt.savefig(f"plots/percentile_plot_{g}.svg")
