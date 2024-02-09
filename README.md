# Combining a dynamic model with a learned model for Planar Quadrotor control
This repository implements the following:
* A simulator for a planar quadrotor system. The simulation only includes gravity, as well as two forces acting on the PQ, one on each rotor, perpendicular to the length of the PQ.
* An optimizer, utlizing the CEM algorithm to control the PQ and lead it to a specified position.

## Simulation
### Simulator
The simulator is located at `src/sim/PlanarQuadrotor.hpp`. The `PlanarQuadrotor` class initializes a new PQ with the given mass, moment of inertia and length. The PQ's initial position is $(0, 0)$.

Using the `update` method, the system is integrated for a given time step, using the provided controls (forces for each rotor). The controls are considered constant for that time step.

The `get_state` method returns the current state of the PQ. The state is a 6D vector, which consists of the linear ($x$, $y$) and angular ($w$) position, as well as the respective velocities. Since acceleration is constant and directly related to the controls passed at each time step, it is not part of the state.

The `get_sim_time` method returns the total time the simulation has been integrated for. The `get_last_ddq` method returns the acceleration computed during the previous integration step.

### Visualizer
The visualizer class, located at `src/sim/Visualizer.hpp`, can be used to visualize the PQ in a (linux) command-line enviroment. The `show` method accepts a `PlanarQuadrotor` object, as well as a pair of integers representing the target position ($x_t$, $y_t$). It then prints the equivalent frame on the screen. An arrow is used to represent the PQ, pointing towards the direction perpendicular to the PQ's length. The target position is represented by the letter T.

The `set_message` method can be used to set a specific message to be printed at the bottom of the screen, for every subsequent frame.

## Optimization
### Dynamic model
To determine the controls required for the PQ to move towards the target position, we use MPC, utilizing the CEM algorithm, implemented in the linked `algevo` library. This repository defines the `ControlIndividual` struct. This consists of a vector of $2h$ values, where $h$ is the horizon, a defined parameter. The struct also defines the method by which each individual is evaluated: the "simulation" is run for $h$ steps and the error is calculated at each step. Consequently, the individual's fitness is relative to the error at each of the $h$ steps.

The algorithm runs iteratively and, once finished, provides the individual with the best overall fitness value. Of this individual, we only use the first two values ($c_{11}$ and $c_{12}$). In this way, the optimizer is able to take into account multiple steps into the future, but we only follow the best initial move, so that we can recalculate the next best move on the next integration step.

In order to run this simulation at each time step, for each individual, we need a model: a method for calculating the accelerations at that time step, based on the input controls. For the PQ, we can easily derive the following forward dynamics model:

$$\Large
\begin{align}
    \ddot{\textbf{q}} &= \begin{bmatrix}
        \frac{-(c_1+c_2)sin(q_3)}{m}\\\\
        \frac{(c_1+c_2)cos(q_3)}{m} - g\\\\
        \frac{(c_2 - c_1)l}{2I}
    \end{bmatrix}
\end{align}
$$

Where:
* $c_1$ and $c_2$ are the provided controls.
* $q_3$ is the angular position of the PQ.
* $g$ is the gravitational acceleration.
* $m$, $l$ and $I$ are the mass, length and moment of inertia of the PQ, respectively.

Since the simulator and the optimizer use the exact same dynamic model, this works perfectly on its own. As long as an appropriate cost function is used, the optimizer is easily able to calculate the required controls to reach the target position.

Next, it is necessary to mimic a situation in which the dynamic model is either not known, or is too complex to account for. In order to do this, there needs to be a mismatch between the actual dynamic model, used by the simulator, and the dynamic model used by the optimizer to find the optimal controls. There are multiple ways to do this, but in this repository, we alter the mass (and subsequently inertia) used by the optimizer.

With a large enough difference between the two gravitational acceleration values, the controls provided by the optimizer result in movement that leads the PQ far from the desired target position.

### Learned model
Now that the true and used model are different to each other, the predicted state on each time step is different to the true state of the system. This difference can be learned, so that the existing dynamic model can be enhanced to be more accurate to the true model.

More specifically, the input of the learned model needs to be the same as that of the dynamic model: the state of the PQ, as well as the controls used for the next time step. The training target needs to be the difference between the actual acceleration caused and the predicted acceleration of the existing dynamic model.

$$
    \begin{align}
        \textbf{x} &= \begin{bmatrix}
            \textbf{q}\\
            \dot{\textbf{q}}\\
            c_1\\
            c_2
        \end{bmatrix},\
        \textbf{t} = \textbf{a}-\textit{d}(\textbf{x})
    \end{align}
$$

Where:
* $\textbf{x}$ and $\textbf{t}$ are the input and desired output, respectively.
* $\textbf{q}$ is the position vector.
* $\dot{\textbf{q}}$ is the velocity vector.
* $c_1$ and $c_2$ are the controls.
* $\textbf{a}$ is the actual acceleration caused by integrating using these controls.
* $d(\cdot)$ is the dynamic model.

A learned model trained in this way allows us to then express the predicted value for each time step as such:

$$
    \begin{align}
        \ddot{\textbf{q}_p} = \textit{d}(\textbf{x}) + \textit{l}(\textbf{x})
    \end{align}
$$

Where $\textit{l}(\cdot)$ is the learned model.

Note that, as seen in $(1)$, the dynamic model only uses the angular position value, $q_3$. Thus, the entire state is not actually necessary and, in fact, we would likely have better performance if we only passed $q_3$ as an input to our learned model. However, in a real-world scenario, we would usually not be aware of which parameters of the state are or are not useful. Therefore, we pass the entire state vector and assume that any non-useful parameters will end up not contributing to the result.

In this repository, a neural network is used to learn this difference. We use the linked library `simple_nn` to initialize a neural network and train it using an episodic approach. This consists of running the simulation and optimization as-is for `n` steps, which make up an episode. We then train the neural network using all collected data during the episode. Following this, we repeat the process again for `m` episodes, or until the change in error between episodes is smaller than a specific threshold value.

This entire process is repeated for `k` runs. We do this to collect data for multiple runs and get a mean error value per step per episode.