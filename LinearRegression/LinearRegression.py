import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Generating dummy data
np.random.seed(0) # Setting a random seed for reproducibility
X = 2 * np.random.rand(100, 1) # Generate a 100x1 array with elements uniformly distributed in [0, 1) multiplied by 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Generating the labels with some noise drawn from a standard normal distribution

# Plotting initial data points
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Data points")
plt.title("Initial Dataset Visualization")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.savefig("LR_initial-data-points.png")

# Initial regression line (before training)
X_line = np.array([[0], [2]])  # Two points for the line
X_line_b = np.c_[np.ones((2, 1)), X_line]  # Add bias term (X_0 = 1)
initial_theta = np.random.randn(2, 1)  # Random initialization of theta
y_line = X_line_b @ initial_theta  # Initial predictions

plt.title("Initial Regression Line")
plt.plot(X_line, y_line, color="orange", label="Initial regression line")
plt.legend()
plt.savefig("LR_Initial-Regression-Line.png")

# Add bias term (X_0 = 1)
X_b = np.c_[np.ones((100, 1)), X]

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

# Initialize parameters (theta)
theta = np.random.randn(2, 1)  # Random initialization

# Gradient Descent
cost_history = []

def compute_cost(X, y, theta):
    """Compute the Mean Squared Error (MSE)."""
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

# Training loop
for iteration in range(n_iterations):
    gradients = (1 / m) * X_b.T @ (X_b @ theta - y)
    theta -= learning_rate * gradients

    cost = compute_cost(X_b, y, theta)
    cost_history.append(cost)

    # Verbose updates
    if iteration % 100 == 0 or iteration == n_iterations - 1:
        print(
            f"Iteration {iteration}: "
            f"Cost = {cost:.4f}, "
            f"Theta0 = {theta[0][0]:.4f}, Theta1 = {theta[1][0]:.4f}"
        )

# Visualizing cost reduction over time
plt.figure(figsize=(10, 5))
plt.plot(range(n_iterations), cost_history, color="purple")
plt.title("Cost Reduction Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.savefig("LR_Cost-Reduction.png")

# Visualizing the final regression line
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label="Data points")
plt.plot(X, X_b @ theta, color="red", label="Final regression line")
plt.title("Final Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.savefig("LR_Final-Regression-Line.png")

print(f"Final Parameters: Intercept = {theta[0][0]:.4f}, Slope = {theta[1][0]:.4f}")


############################# VISUALIZING THE CURVE OF MSE #############################
# Create a grid of theta values
theta0_vals = np.linspace(theta[0][0] - 5, theta[0][0] + 5, 100)
theta1_vals = np.linspace(theta[1][0] - 5, theta[1][0] + 5, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

# Compute the MSE for each combination of theta0 and theta1
mse_vals = np.zeros_like(theta0_grid)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta_temp = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        mse_vals[i, j] = compute_cost(X_b, y, theta_temp)

# Plot the 3D surface of the MSE
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_grid, theta1_grid, mse_vals, cmap='viridis', alpha=0.8)
ax.set_xlabel('Theta0 (Intercept)')
ax.set_ylabel('Theta1 (Slope)')
ax.set_zlabel('MSE')
ax.set_title('MSE as a Function of Theta0 and Theta1')
plt.savefig("LR_MSE_Parabolic_Curve.png")

############################# END OF VISUALIZATION #############################