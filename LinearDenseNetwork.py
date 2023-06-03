#I only took Nairis code and added activation functions

import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation='none'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        self.activation = activation
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.act_funct(np.dot(inputs, self.weights) + self.biases)
        
        return self.output
    
    def backward(self, grad_output, learning_rate):
        grad_activation = self.act_der(self.output)*grad_output
        grad_weights = np.dot(self.inputs.T, grad_activation)
        grad_biases = np.sum(grad_output, axis=0)
        
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input
    
    def act_funct(self, x):
        self.x = x
        if(self.activation=='none'):
            return x
        if(self.activation=='sigmoid'):
            return 1/(1+np.exp(-x))
        if(self.activation=='relu'):
            if(self.x>0):
                return self.x
            else:
                return 0
    def act_der(self, x):
        self.x = x
        if self.activation=='none':
            return 1
        if(self.activation=='sigmoid'):
            return self.act_funct(x)*(1-self.act_funct(x))
        if(self.activation=='relu'):
            if(x>0):
                return 1
            else:
                return 0
        
    
class DenseNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
    