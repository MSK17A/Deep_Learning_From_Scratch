from turtle import color
import numpy as np
from matplotlib import pyplot as plt
from Computational_Graph import *

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2*np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2*np.ones((50, 2))

# Plot the red and blue points
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0],blue_points[:,1],color='blue')

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()


# Create an interval from -5 to 5 in steps of 0.01
a = np.arange(-5, 5, 0.01)

# Compute corresponding sigmoid function values
s = 1 / (1 + np.exp(-a))

# Plot them
plt.plot(a, s)
plt.grid(True)
plt.show()

# Implement sigmoid function as an operation class
class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, z):
        """Construct sigmoid

        Args:
            z: Input node
        """
        super().__init__([z])
    def compute(self, z_value):
        """Compute the output of the sigmoid operation

        Args:
            z_value: Input value
        """
        return 1 / (1+np.exp(-z_value))

# Example
# Create a new graph
Graph().as_default()

x = placeholder()
w = Variable([1, 1])
b = Variable(0)
p = sigmoid( add(matmul(w, x), b) )

session = Session()
print(session.run(p, {
    x: [3, 2]
}))

class softmax(Operation):
    """Returns the softmax of z.
    """

    def __init__(self, z):
        """Construct softmax

        Args:
          z: Input node
        """
        super().__init__([z])

    def compute(self, z_value):
        """Compute the output of the softmax operation

        Args:
          z_value: Input value
        """
        return np.exp(z_value) / np.sum(np.exp(z_value), axis=1)[:, None]


# Example
# Create a new graph
Graph().as_default()

X = placeholder()

# Create a weight matrix for 2 output classes:
# One with a weight vector (1, 1) for blue and one for a weight (-1, -1) for red
W = Variable([
    [1,-1],
    [1,-1]
])
b = Variable([0,0])
p = softmax( add(matmul(X, W), b) )

# Create a session and run the perception on our blue/red points
session = Session()
output_probabilities = session.run(p, {
    X: np.concatenate((blue_points, red_points))
})

# Print the first 10 lines, corresponding to the probabilities of the first 10 points
print(output_probabilities[:10])