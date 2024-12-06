import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Parameters
dt = 0.1  # Time step
N = 10  # Prediction horizon
max_speed = 1.0  # Maximum speed
obstacle_center = np.array([2.0, 2.0])  # Obstacle center
obstacle_radius = 0.5  # Obstacle radius
goal = np.array([4.0, 4.0])  # Goal position

# Dynamics: x_{k+1} = Ax_k + Bu_k
A = np.eye(2)
B = dt * np.eye(2)

# Initial state
x0 = np.array([0.0, 0.0])  # Start position

# Cost weights
Q = np.eye(2)  # State cost
R = 0.01 * np.eye(2)  # Control effort cost

# Number of simulation steps
sim_steps = 50

# Storage for trajectory
trajectory = [x0]

# MPC loop
x = x0
for _ in range(sim_steps):
    # Define optimization variables
    X = cp.Variable((2, N + 1))  # States
    U = cp.Variable((2, N))  # Control inputs

    # Define cost function
    cost = 0
    constraints = []
    for k in range(N):
        cost += cp.quad_form(X[:, k] - goal, Q) + cp.quad_form(U[:, k], R)
        constraints += [X[:, k + 1] == A @ X[:, k] + B @ U[:, k]]
        constraints += [cp.norm(U[:, k], 'inf') <= max_speed]
        # Obstacle avoidance constraint (corrected element-wise multiplication)
        constraints += [
            (X[0, k] - obstacle_center[0]) * (X[0, k] - obstacle_center[0]) +
            (X[1, k] - obstacle_center[1]) * (X[1, k] - obstacle_center[1])
            >= obstacle_radius ** 2
        ]

    # Terminal cost
    cost += cp.quad_form(X[:, N] - goal, Q)

    # Initial condition constraint
    constraints += [X[:, 0] == x]

    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        print("Optimization problem could not be solved!")
        break

    # Apply the first control input
    u = U[:, 0].value
    x = A @ x + B @ u
    trajectory.append(x)

    # Break if goal is reached
    if np.linalg.norm(x - goal) < 0.1:
        print("Goal reached!")
        break

# Convert trajectory to a numpy array
trajectory = np.array(trajectory)

# Plot the trajectory and obstacle
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', label="Trajectory")
plt.scatter(*goal, color='green', label="Goal")
circle = plt.Circle(obstacle_center, obstacle_radius, color='red', alpha=0.5, label="Obstacle")
plt.gca().add_artist(circle)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("MPC Navigation Around Obstacle")
plt.show()
