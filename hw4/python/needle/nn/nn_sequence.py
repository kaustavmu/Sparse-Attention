"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.exp(x) / (1 + ops.exp(x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        
        # Initialize weights
        k = 1 / hidden_size
        bound = np.sqrt(k)
        
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype))

    def forward(self, X, h=None):
        bs = X.shape[0]
        
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        
        # Compute linear transformation
        linear = X @ self.W_ih + h @ self.W_hh
        
        if self.bias:
            # Reshape bias to be broadcastable: (hidden_size,) -> (1, hidden_size)
            bias_ih_reshaped = ops.reshape(self.bias_ih, (1, self.hidden_size))
            bias_hh_reshaped = ops.reshape(self.bias_hh, (1, self.hidden_size))
            
            # Broadcast to match linear shape
            bias_ih_broadcasted = ops.broadcast_to(bias_ih_reshaped, linear.shape)
            bias_hh_broadcasted = ops.broadcast_to(bias_hh_reshaped, linear.shape)
            
            linear = linear + bias_ih_broadcasted + bias_hh_broadcasted
        
        # Apply nonlinearity
        if self.nonlinearity == 'tanh':
            return ops.tanh(linear)
        elif self.nonlinearity == 'relu':
            return ops.relu(linear)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn_cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            cell = RNNCell(cell_input_size, hidden_size, bias, nonlinearity, device, dtype)
            self.rnn_cells.append(cell)

    def forward(self, X, h0=None):
        seq_len, bs, _ = X.shape
        
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        
        # Split h0 along the layer dimension to get individual layer states
        # h0 shape: (num_layers, bs, hidden_size)
        if self.num_layers == 1:
            # For single layer, just squeeze the first dimension
            h_states = [ops.reshape(h0, (bs, self.hidden_size))]
        else:
            # Split along axis 0 (layer dimension)
            h_list = ops.split(h0, axis=0)
            h_states = []
            for h in h_list:
                # Each h has shape (1, bs, hidden_size), reshape to (bs, hidden_size)
                h_states.append(ops.reshape(h, (bs, self.hidden_size)))
        
        outputs = []
        
        # Split X along the sequence dimension
        # X shape: (seq_len, bs, input_size)
        x_list = ops.split(X, axis=0)
        
        for x_t_with_dim in x_list:
            # x_t_with_dim has shape (1, bs, input_size), reshape to (bs, input_size)
            x_t = ops.reshape(x_t_with_dim, (bs, x_t_with_dim.shape[-1]))
            
            # Pass through each layer
            for layer in range(self.num_layers):
                h_states[layer] = self.rnn_cells[layer](x_t, h_states[layer])
                x_t = h_states[layer]  # Output of this layer becomes input to next
            
            outputs.append(h_states[-1])  # Save output from last layer
        
        # Stack outputs: convert list of (bs, hidden_size) to (seq_len, bs, hidden_size)
        output = ops.stack(outputs, axis=0)
        
        # Stack final hidden states: convert list to (num_layers, bs, hidden_size)
        h_n = ops.stack(h_states, axis=0)
        
        return output, h_n

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Initialize weights
        k = 1 / hidden_size
        bound = np.sqrt(k)
        
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))

    def forward(self, X, h=None):
        bs = X.shape[0]
        
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        
        # Compute gates
        gates = X @ self.W_ih + h0 @ self.W_hh
        
        if self.bias:
            # Proper bias handling like in RNNCell
            bias_ih_reshaped = ops.reshape(self.bias_ih, (1, 4 * self.hidden_size))
            bias_hh_reshaped = ops.reshape(self.bias_hh, (1, 4 * self.hidden_size))
            bias_ih_broadcasted = ops.broadcast_to(bias_ih_reshaped, gates.shape)
            bias_hh_broadcasted = ops.broadcast_to(bias_hh_reshaped, gates.shape)
            gates = gates + bias_ih_broadcasted + bias_hh_broadcasted
        
        # Split gates using slicing
        i_gate = Sigmoid()(ops.get_item(gates, (slice(None), slice(None, self.hidden_size))))
        f_gate = Sigmoid()(ops.get_item(gates, (slice(None), slice(self.hidden_size, 2*self.hidden_size))))
        g_gate = ops.tanh(ops.get_item(gates, (slice(None), slice(2*self.hidden_size, 3*self.hidden_size))))
        o_gate = Sigmoid()(ops.get_item(gates, (slice(None), slice(3*self.hidden_size, None))))
        
        # Update cell and hidden states
        c_new = f_gate * c0 + i_gate * g_gate
        h_new = o_gate * ops.tanh(c_new)
        
        return h_new, c_new


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(cell_input_size, hidden_size, bias, device, dtype)
            self.lstm_cells.append(cell)

    def forward(self, X, h=None):
        seq_len, bs, _ = X.shape
        
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        
        # Split h0 and c0 along the layer dimension like in RNN
        if self.num_layers == 1:
            h_states = [ops.reshape(h0, (bs, self.hidden_size))]
            c_states = [ops.reshape(c0, (bs, self.hidden_size))]
        else:
            h_list = ops.split(h0, axis=0)
            c_list = ops.split(c0, axis=0)
            h_states = []
            c_states = []
            for h, c in zip(h_list, c_list):
                h_states.append(ops.reshape(h, (bs, self.hidden_size)))
                c_states.append(ops.reshape(c, (bs, self.hidden_size)))
        
        outputs = []
        
        # Split X along the sequence dimension like in RNN
        x_list = ops.split(X, axis=0)
        
        for x_t_with_dim in x_list:
            # x_t_with_dim has shape (1, bs, input_size), reshape to (bs, input_size)
            x_t = ops.reshape(x_t_with_dim, (bs, x_t_with_dim.shape[-1]))
            
            # Pass through each layer
            for layer in range(self.num_layers):
                h_states[layer], c_states[layer] = self.lstm_cells[layer](x_t, (h_states[layer], c_states[layer]))
                x_t = h_states[layer]  # Output becomes input to next layer
            
            outputs.append(h_states[-1])  # Save output from last layer
        
        # Stack outputs and final states
        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h_states, axis=0)
        c_n = ops.stack(c_states, axis=0)
        
        return output, (h_n, c_n)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        seq_len, bs = x.shape
        # Convert indices to one-hot vectors
        x_one_hot = init.one_hot(self.num_embeddings, x, device=x.device)
        # Reshape for matrix multiplication
        x_one_hot = ops.reshape(x_one_hot, (seq_len * bs, self.num_embeddings))
        # Apply embedding transformation
        output = x_one_hot @ self.weight
        # Reshape back to (seq_len, bs, embedding_dim)
        output = ops.reshape(output, (seq_len, bs, self.embedding_dim))
        return output
