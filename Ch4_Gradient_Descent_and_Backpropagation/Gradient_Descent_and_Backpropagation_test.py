from Training_Critirion_test import *
from queue import Queue

class GradientDescentOptimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    
    def minimize(self, loss):

        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):

                grad_table = compute_gradients(loss)
                
                for node in grad_table:
                    if type(node) == Variable:

                        grad = grad_table[node]

                        node.value -= learning_rate * grad
        return MinimizationOperation()

# A dictionary that will map operations to gradient functions
_gradient_registry = {}

class RegisterGradient:
    """A decorator for registering the gradient function for an op type.
    """

    def __init__(self, op_type):
        """Creates a new decorator with `op_type` as the Operation type.
        Args:
          op_type: The name of an operation
        """
        self._op_type = eval(op_type)

    def __call__(self, f):
        """Registers the function `f` as gradient function for `op_type`."""
        _gradient_registry[self._op_type] = f
        return f

from queue import Queue


def compute_gradients(loss):
    # grad_table[node] will contain the gradient of the loss w.r.t. the node's output
    grad_table = {}

    # The gradient of the loss with respect to the loss is just 1
    grad_table[loss] = 1

    # Perform a breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # If this node is not the loss
        if node != loss:
            #
            # Compute the gradient of the loss with respect to this node's output
            #
            grad_table[node] = 0

            # Iterate all consumers
            for consumer in node.consumers:

                # Retrieve the gradient of the loss w.r.t. consumer's output
                lossgrad_wrt_consumer_output = grad_table[consumer]

                # Retrieve the function which computes gradients with respect to
                # consumer's inputs given gradients with respect to consumer's output.
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]

                # Get the gradient of the loss with respect to all of consumer's inputs
                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    # If there is a single input node to the consumer, lossgrads_wrt_consumer_inputs is a scalar
                    grad_table[node] += lossgrads_wrt_consumer_inputs

                else:
                    # Otherwise, lossgrads_wrt_consumer_inputs is an array of gradients for each input node

                    # Retrieve the index of node in consumer's inputs
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)

                    # Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    # Add to total gradient
                    grad_table[node] += lossgrad_wrt_node

        #
        # Append each input node to the queue
        #
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    # Return gradients for each visited node
    return grad_table

@RegisterGradient("negative")
def _negative_gradient(op, grad):
    """Computes the gradients for 'negative'

    Args:
        op: The 'negative' 'Operation' that we are diffrentiating
        grad: Gradient with respect ot the output of the 'negative' op
    """
    return -grad

@RegisterGradient("log")
def _log_gradient(op, grad):
    """Computes the gradient for 'log'

    Args:
      op: The `log` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `log` op.

    Returns:
        Gradients with respect to the input of 'log'.
    """
    x = op.inputs[0]
    return grad/x

@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    """Computes the gradients for `sigmoid`.

    Args:
      op: The `sigmoid` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `sigmoid` op.

    Returns:
      Gradients with respect to the input of `sigmoid`.
    """

    sigmoid = op.output

    return grad * sigmoid * (1 - sigmoid)

@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    """Computes the gradients for `multiply`.

    Args:
      op: The `multiply` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `multiply` op.

    Returns:
      Gradients with respect to the input of `multiply`.
    """

    A = op.inputs[0]
    B = op.inputs[1]

    return [grad * B, grad * A]

@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    """Computes the gradients for `matmul`.

    Args:
      op: The `matmul` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `matmul` op.

    Returns:
      Gradients with respect to the input of `matmul`.
    """

    A = op.inputs[0]
    B = op.inputs[1]

    return [grad.dot(B.T), A.T.dot(grad)]

@RegisterGradient("add")
def _add_gradient(op, grad):
    """Computes the gradients for `add`.

    Args:
      op: The `add` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `add` op.

    Returns:
      Gradients with respect to the input of `add`.
    """
    a = op.inputs[0]
    b = op.inputs[1]

    grad_wrt_a = grad
    grad_wrt_b = grad

    #
    # The following becomes relevant if a and b are of different shapes.
    #
    while np.ndim(grad_wrt_a) > len(a.shape):
        grad_wrt_a = np.sum(grad_wrt_a, axis=0)
    for axis, size in enumerate(a.shape):
        if size == 1:
            grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)

    while np.ndim(grad_wrt_b) > len(b.shape):
        grad_wrt_b = np.sum(grad_wrt_b, axis=0)
    for axis, size in enumerate(b.shape):
        if size == 1:
            grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)

    return [grad_wrt_a, grad_wrt_b]

@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    """Computes the gradients for `reduce_sum`.

    Args:
      op: The `reduce_sum` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `reduce_sum` op.

    Returns:
      Gradients with respect to the input of `reduce_sum`.
    """
    A = op.inputs[0]

    output_shape = np.array(A.shape)
    output_shape[op.axis] = 1
    tile_scaling = A.shape // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)

@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    """Computes the gradients for `softmax`.

    Args:
      op: The `softmax` `Operation` that we are differentiating
      grad: Gradient with respect to the output of the `softmax` op.

    Returns:
      Gradients with respect to the input of `softmax`.
    """

    softmax = op.output
    return (grad - np.reshape(
        np.sum(grad * softmax, 1),
        [-1, 1]
    )) * softmax

# Create a new graph
Graph().as_default()

X = placeholder()
c = placeholder()

# Initialize weights randomly
W = Variable(np.random.randn(2, 2))
b = Variable(np.random.randn(2))

# Build perceptron
p = softmax(add(matmul(X, W), b))

# Build cross-entropy loss
J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

# Build minimization op
minimization_op = GradientDescentOptimizer(learning_rate=0.01).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

# Create session
session = Session()

# Perform 100 gradient descent steps
for step in range(1000):
    J_value = session.run(J, feed_dict)
    if step % 10 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)

# Print final result
W_value = session.run(W)
print("Weight matrix:\n", W_value)
b_value = session.run(b)
print("Bias:\n", b_value)