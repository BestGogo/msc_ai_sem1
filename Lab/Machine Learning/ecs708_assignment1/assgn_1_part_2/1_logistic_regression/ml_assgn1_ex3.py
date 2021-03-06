from load_data_ex1 import *
from normalize_features import *
from gradient_descent import *
from plot_data_function import *
from plot_boundary import *
import matplotlib.pyplot as plt
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex1()

X = np.append(X, np.square(X),axis =1)
X = np.append(X, (X[:,1]*X[:,0]).reshape(X.shape[0],1),axis =1)

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X_normalized, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)


theta = np.zeros((6))
# Set learning rate alpha and number of iterations
alpha = 1.0
iterations = 100

# Call the gradient descent function to obtain the trained parameters theta_final and the cost vector
theta_final, cost_vector = gradient_descent(X_normalized, y, theta, alpha, iterations)

# Plot the cost for all iterations
fig, ax1 = plt.subplots()
plot_cost(cost_vector, ax1)
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex3_cost.png')
plt.savefig(plot_filename)
min_cost = np.min(cost_vector)
argmin_cost = np.argmin(cost_vector)
print('Final cost: {:.5f}'.format(cost_vector[-1]))
print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))

# enter non-interactive mode of matplotlib, to keep figures open
fig, ax1 = plt.subplots()
ax1 = plot_data_function(X_normalized, y, ax1)
plot_boundary(X_normalized, theta_final, ax1)
# save the plotted decision boundary as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'ex3_decision_boundary.png')
plt.savefig(plot_filename)

# plt.ioff()
# plt.show()
