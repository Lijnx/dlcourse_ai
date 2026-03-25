import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W * W)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    probs = np.atleast_2d(predictions)

    probs = probs - np.max(probs, axis=1, keepdims=True)
    probs = np.exp(probs)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    probs_2d = np.atleast_2d(probs)
    target_1d = np.atleast_1d(target_index)

    correct_logprobs = -np.log(probs_2d[np.arange(probs_2d.shape[0]), target_1d])
    return np.mean(correct_logprobs)


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)

    t = np.atleast_1d(target_index)
    d_preds = np.atleast_2d(probs)
    d_preds[np.arange(d_preds.shape[0]), t.flatten()] -= 1
    d_preds /= d_preds.shape[0]

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(X < 0, 0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * (self.X > 0)

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        if self.padding > 0:
            X = np.pad(
                X,
                ((0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0)),
                mode='constant'
            )
        
        self.X = X
        K = self.filter_size
        out_height = height - K + 2*self.padding + 1
        out_width = width - K + 2*self.padding + 1
        
        # Получаем все окна сразу
        # Было: (BS, H, W, Cin)
        # Станет: (BS, OH, OW, K, K, Cin)
        windows = sliding_window_view(X, window_shape=(K, K), axis=(1, 2))
        windows = np.transpose(windows, (0, 1, 2, 4, 5, 3))

        # Разворачиваем каждое окно в вектор длины K*K*Cin
        windows_flat = windows.reshape(batch_size, out_height, out_width, -1)
        # shape: (BS, OH, OW, K*K*Cin)

        # Разворачиваем фильтры
        W_flat = self.W.value.reshape(-1, self.out_channels)
        # shape: (K*K*Cin, Cout)

        # Одно матричное умножение для всех batch, y, x
        result = windows_flat @ W_flat
        # shape: (BS, OH, OW, Cout)

        # Добавляем bias
        result = result + self.B.value
        # self.B.value shape: (Cout,), забродкастится по (BS, OH, OW)

        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        K = self.filter_size
        _, out_height, out_width, out_channels = d_out.shape # (BS, OH, OW, Cout)

        windows = sliding_window_view(self.X, window_shape=(K, K), axis=(1, 2))
        windows = np.transpose(windows, (0, 1, 2, 4, 5, 3))
        windows = windows.reshape(-1, K * K * channels)  # (BS*OH*OW, K*K*Cin)
        
        grad_flat = d_out.reshape(-1, out_channels) # (BS*OH*OW, Cout)

        self.W.grad += (windows.T @ grad_flat).reshape(K, K, channels, out_channels)
        self.B.grad += np.sum(grad_flat, axis=0)

        # Gradient by input
        W_flat = self.W.value.reshape(-1, out_channels) # (K*K*Cin, Cout)
        dX = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                grad_here = d_out[:, y, x, :] @ W_flat.T  # (BS, K*K*Cin)
                grad_here = grad_here.reshape(batch_size, K, K, channels)

                dX[:, y:y+K, x:x+K, :] += grad_here

        if self.padding > 0:
            dX = dX[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return dX


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        K = self.pool_size
        S = self.stride
        self.X = X

        windows = sliding_window_view(X, window_shape=(K, K), axis=(1, 2))
        windows = windows[:, ::S, ::S, :, :, :]
        _, out_height, out_width, _, _, _ = windows.shape

        windows_flat = windows.reshape(batch_size, out_height, out_width, channels, -1)

        result = np.max(windows_flat, axis=4)

        return result

    def backward(self, d_out):
        # (BS, OH, OW, Cin)
        batch_size, height, width, channels = self.X.shape
        K = self.pool_size
        S = self.stride

        dX = np.zeros_like(self.X)

        # (BS, OH, OW, Cin, K, K)
        windows = sliding_window_view(self.X, window_shape=(K, K), axis=(1, 2)) 
        windows = windows[:, ::S, ::S, :, :, :]
        _, out_height, out_width, _, _, _ = windows.shape

        # (BS, OH, OW, Cin, K*K)
        windows_flat = windows.reshape(batch_size, out_height, out_width, channels, K * K)
        max_ids = np.argmax(windows_flat, axis=4)

        # Переводим плоский индекс в координаты внутри окна
        # shape: (BS, OH, OW, Cin)
        ky = max_ids // K
        kx = max_ids % K

        # Индексы для batch / output-position / channel
        b_idx = np.arange(batch_size)[:, None, None, None]   # (BS, 1, 1, 1)
        y_idx = np.arange(out_height)[None, :, None, None]   # (1, OH, 1, 1)
        x_idx = np.arange(out_width)[None, None, :, None]    # (1, 1, OW, 1)
        c_idx = np.arange(channels)[None, None, None, :]     # (1, 1, 1, Cin)

        # Координаты во входе, куда надо положить градиент
        in_y = y_idx * S + ky
        in_x = x_idx * S + kx

        # Разбрасываем d_out обратно в dX
        np.add.at(dX, (b_idx, in_y, in_x, c_idx), d_out)
            
        return dX

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)


    def params(self):
        # No params!
        return {}
