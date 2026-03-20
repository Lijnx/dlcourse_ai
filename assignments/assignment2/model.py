import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.hidden = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.output = FullyConnectedLayer(hidden_layer_size, n_output)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        params = self.params()

        for param in params.values():
            param.grad = np.zeros_like(param.grad)

        logits = self.output.forward( self.relu.forward( self.hidden.forward(X) ) )
        loss, grad = softmax_with_cross_entropy(logits, y)

        grad = self.output.backward(grad)
        grad = self.relu.backward(grad)
        self.hidden.backward(grad)


        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for name, param in params.items():
            if name.endswith('_W'):
                reg_loss, reg_grad = l2_regularization(param.value, self.reg)
                loss += reg_loss
                param.grad += reg_grad


        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = np.zeros(X.shape[0], np.int64)

        logits = self.output.forward( self.relu.forward( self.hidden.forward(X) ) )
        pred = np.argmax(logits, axis=1)

        return pred

    def params(self):
        result = {}

        result = {'hidden_W': self.hidden.W, 'hidden_B': self.hidden.B, 
                  'output_W': self.output.W, 'output_B': self.output.B}

        return result
