import numpy as np
from train_scripts import *
from plot_cost import *
import matplotlib.pyplot as plt

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]
              ])

y_XOR = np.array([0, 1, 1, 0])
y_NOR = np.array([1, 0, 0, 0])
y_AND = np.array([0, 0 ,0 ,1])
y = y_AND

n_hidden = 2
iterations = 10000
learning_rate = 0.1

# alpha = 0.1
# Sample #01 | Target value: 0.00 | Predicted value: 0.05508
# Sample #02 | Target value: 1.00 | Predicted value: 0.94994
# Sample #03 | Target value: 1.00 | Predicted value: 0.94981
# Sample #04 | Target value: 0.00 | Predicted value: 0.05371
# Minimum cost: 0.00548, on iteration #10000




# Train the neural network on the XOR problem
# For now, we will not use the 3rd and 4th outputs of the function, hence we use "_" on the returned outputs
errors, nn, _, _ = train(X, y, n_hidden, iterations, learning_rate)
# Test the neural network on the XOR problem
test_xor(X, y, nn)

# Plot the cost for all iterations
fig, ax1 = plt.subplots()
plot_cost(errors, ax1)
plot_filename = os.path.join(os.getcwd(), 'figures', 'XOR_cost.png')
plt.savefig(plot_filename)
min_cost = np.min(errors)
argmin_cost = np.argmin(errors)
print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
