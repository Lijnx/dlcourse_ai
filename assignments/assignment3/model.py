import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        filter1_size = 3
        padding1 = 0

        filter2_size = 3
        padding2 = 0

        maxpool1_size = 4
        stride1 = 1
        maxpool2_size = 4
        stride2 = 1

        w, h, n_channels = input_shape

        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, filter_size=filter1_size, padding=0)
        w = w - filter1_size + 2 * padding1 + 1
        h = h - filter1_size + 2 * padding1 + 1
        c = conv1_channels

        self.relu1 = ReLULayer()

        self.maxpool1 = MaxPoolingLayer(maxpool1_size, stride=stride1)
        w = (w - maxpool1_size) // stride1 + 1
        h = (h - maxpool1_size) // stride1 + 1

        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=filter2_size, padding=0)
        w = w - filter2_size + 2 * padding2 + 1
        h = h - filter2_size + 2 * padding2 + 1
        c = conv2_channels

        self.relu2 = ReLULayer()

        self.maxpool2 = MaxPoolingLayer(maxpool2_size, stride=stride2)
        w = (w - maxpool2_size) // stride2 + 1
        h = (h - maxpool2_size) // stride2 + 1

        self.flattener = Flattener()
        
        fc_in = h * w * c
        self.fc = FullyConnectedLayer(fc_in, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        for param in self.params().values():
            param.grad.fill(0)

        conv1_out = self.conv1.forward(X)
        relu1_out = self.relu1.forward(conv1_out)
        maxpool1_out = self.maxpool1.forward(relu1_out)
        conv2_out = self.conv2.forward(maxpool1_out)
        relu2_out = self.relu2.forward(conv2_out)
        maxpool2_out = self.maxpool2.forward(relu2_out)
        flat_out = self.flattener.forward(maxpool2_out)
        logits = self.fc.forward(flat_out)

        loss, grad = softmax_with_cross_entropy(logits, y)

        grad_fc = self.fc.backward(grad)
        grad_flat = self.flattener.backward(grad_fc)
        grad_maxpool2 = self.maxpool2.backward(grad_flat)
        grad_relu2 = self.relu2.backward(grad_maxpool2)
        grad_conv2 = self.conv2.backward(grad_relu2)
        grad_maxpool1 = self.maxpool1.backward(grad_conv2)
        grad_relu1 = self.relu1.backward(grad_maxpool1)
        self.conv1.backward(grad_relu1)

        return loss

    def predict(self, X):
        pred = np.zeros(X.shape[0], np.int64)

        conv1_out = self.conv1.forward(X)
        relu1_out = self.relu1.forward(conv1_out)
        maxpool1_out = self.maxpool1.forward(relu1_out)
        conv2_out = self.conv2.forward(maxpool1_out)
        relu2_out = self.relu2.forward(conv2_out)
        maxpool2_out = self.maxpool2.forward(relu2_out)
        flat_out = self.flattener.forward(maxpool2_out)
        logits = self.fc.forward(flat_out)

        pred = np.argmax(logits, axis=1)

        return pred

    def params(self):
        
        result = {'conv1_W': self.conv1.W, 'conv1_B': self.conv1.B, 
                  'conv2_W': self.conv2.W, 'conv2_B': self.conv2.B,
                  'fc_W'   : self.fc.W,    'fc_B'   : self.fc.B
                  }

        return result
