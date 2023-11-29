from os import mkdir
from os.path import isdir
from sys import argv
import matplotlib.pyplot as plt

# Make directory to save plots
if not isdir("plots"):
    mkdir("plots")

# Read data from file
with open(argv[1], "r") as f:
    data = f.readlines()

# Parse parameters
params = data[0].split()
g = float(params[0])
steps = int(params[1])
episodes = int(params[2])
runs = int(params[3])
target = (float(params[4]), float(params[5]))

# Parse data
data = data[1:]
# Keep a sum for each step of each episode
episode_data = [[0]*steps for _ in range(episodes)]
# For each run, sum the data for each step of each episode
for i in range(runs):
    run_idx = i*episodes
    for j in range(episodes):
        l = data[run_idx + j].split()
        for k in range(steps):
            episode_data[j][k] += float(l[k])

# Average the data for each step of each episode
for i in range(episodes):
    for j in range(steps):
        episode_data[i][j] /= runs

# Create one plot for each episode
for i in range(episodes):
    plt.plot(episode_data[i], label=f"Episode {i+1}")
    plt.xlabel("Steps")
    plt.ylabel("Error")
    plt.title(f"g = {g}")
    plt.legend()
    plt.savefig(f"plots/{g}_{i+1}.png")
    plt.clf()
