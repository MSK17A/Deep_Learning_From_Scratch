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

