import numpy as np
import matplotlib.pyplot as plt
import random

# Define parameters
x_start = np.array([0, 0, 0, 0])  # Initial state: [x, y, vx, vy]
x_goal = np.array([10, 10, 0, 0])  # Goal state: [x, y, vx, vy]
obstacles = [((5, 5), 1), ((7, 8), 2)]  # Obstacles: (center, radius)
max_iters = 500  # Maximum iterations
step_size = 1.0  # Step size for the RRT
goal_sample_rate = 0.1  # Probability of sampling the goal
safety_margin = 0.2  # Safety margin around obstacles

# Helper function: Check collision with obstacles
def is_collision_free_with_margin(x, obstacles, margin):
    for (cx, cy), r in obstacles:
        if np.linalg.norm(np.array([x[0], x[1]]) - np.array([cx, cy])) < r + margin:
            return False
    return True

# RRT algorithm
def rrt_with_margin(start, goal, obstacles, max_iters, step_size, goal_sample_rate, margin):
    nodes = [start]  # Tree nodes
    edges = []  # Connections between nodes

    for _ in range(max_iters):
        # Randomly sample a point (biased towards goal)
        if random.random() < goal_sample_rate:
            sample = goal[:2]
        else:
            sample = np.random.uniform(low=[-1, -1], high=[12, 12], size=2)

        # Find the nearest node in the tree
        nearest_idx = np.argmin([np.linalg.norm(np.array(node[:2]) - sample) for node in nodes])
        nearest_node = nodes[nearest_idx]

        # Steer towards the sample point
        direction = (sample - np.array(nearest_node[:2]))
        if np.linalg.norm(direction) > step_size:
            direction = direction / np.linalg.norm(direction) * step_size
        new_node = nearest_node[:2] + direction
        new_node = np.append(new_node, [0, 0])  # Add velocity components for consistency

        # Check for collisions with safety margin
        if not is_collision_free_with_margin(new_node, obstacles, margin):
            continue

        # Add the new node and edge to the tree
        nodes.append(new_node)
        edges.append((nearest_idx, len(nodes) - 1))

        # Check if the goal is reached
        if np.linalg.norm(new_node[:2] - goal[:2]) < step_size:
            nodes.append(goal)
            edges.append((len(nodes) - 2, len(nodes) - 1))
            break

    return nodes, edges

# Run RRT with safety margin
nodes_margin, edges_margin = rrt_with_margin(x_start, x_goal, obstacles, max_iters, step_size, goal_sample_rate, safety_margin)

# Visualize the RRT with margin
fig, ax = plt.subplots(figsize=(10, 10))
# Plot start and goal
ax.scatter(*x_start[:2], color="green", label="Start", s=100)
ax.scatter(*x_goal[:2], color="red", label="Goal", s=100)

# Plot obstacles with margin
for (cx, cy), r in obstacles:
    circle = plt.Circle((cx, cy), r + safety_margin, color="gray", alpha=0.5, linestyle='--', label="Safety Margin" if 'Safety Margin' not in ax.get_legend_handles_labels()[1] else "")
    ax.add_artist(circle)
    circle_actual = plt.Circle((cx, cy), r, color="gray", alpha=0.5)
    ax.add_artist(circle_actual)

# Plot tree
for edge in edges_margin:
    node1 = nodes_margin[edge[0]]
    node2 = nodes_margin[edge[1]]
    ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color="blue", alpha=0.5)

# Plot path
if nodes_margin[-1][0:2].tolist() == x_goal[0:2].tolist():  # If the goal was reached
    path = [nodes_margin[-1]]
    parent_idx = edges_margin[-1][0]
    while parent_idx != 0:
        path.append(nodes_margin[parent_idx])
        parent_idx = edges_margin[[e[1] for e in edges_margin].index(parent_idx)][0]
    path.append(x_start)
    path = path[::-1]
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, color="orange", label="Path", linewidth=2)

# Configure plot
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)
ax.legend()
plt.title("RRT Path Planning with Safety Margin")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()
