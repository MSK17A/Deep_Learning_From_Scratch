from Computational_Graph import *

# Create a new graph
G = Graph().as_default()

# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# Create placeholder
x = placeholder()

# Create hidden node y
y = matmul(A, x)

# Create output node z
z = add(y, b)


session = Session()
output = session.run(z, {
    x: [1, 2]
})
print(output)