import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
from needle.nn.nn_transformer_sparse import Transformer as SparseTransformer
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype
        self.model = nn.Sequential(
            self.ConvBN(3, 16, 7, 4, device = self.device, dtype = self.dtype),
            self.ConvBN(16, 32, 3, 2, device = self.device, dtype = self.dtype),
            nn.Residual(
                nn.Sequential(
                    self.ConvBN(32, 32, 3, 1, device = self.device, dtype = self.dtype),
                    self.ConvBN(32, 32, 3, 1, device = self.device, dtype = self.dtype)
                )
            ),
            self.ConvBN(32, 64, 3, 2, device = self.device, dtype = self.dtype),
            self.ConvBN(64, 128, 3, 2, device = self.device, dtype = self.dtype),
            nn.Residual(
                nn.Sequential(
                    self.ConvBN(128, 128, 3, 1, device = self.device, dtype = self.dtype),
                    self.ConvBN(128, 128, 3, 1, device = self.device, dtype = self.dtype)
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, device = self.device, dtype = self.dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device = self.device, dtype = self.dtype)
        )
        ### END YOUR SOLUTION

    def ConvBN(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        """
        Helper function to create a sequence of Conv -> BatchNorm -> ReLU
        with given parameters.
        """
        ### BEGIN YOUR SOLUTION
        return nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype),
            nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
            nn.ReLU()
        )
    
    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=20, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.seq_model = seq_model
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == "rnn":
            self.model = nn.RNN(
                embedding_size,
                hidden_size,
                num_layers,
                device=device,
                dtype=dtype,
            )
        elif seq_model == "lstm":
            self.model = nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers,
                device=device,
                dtype=dtype,
            )
        elif seq_model == "transformer":
            self.model = SparseTransformer(
                embedding_size,
                hidden_size,
                num_layers,
                device=device,
                dtype=dtype,
                sequence_len=seq_len,
            )
        else:
            raise ValueError(f"Unknown seq_model '{seq_model}'")
        self.linear = nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Inputs:
            x: (seq_len, bs)
        Returns:
            out: (seq_len*bs, output_size)
        """
        seq_len, bs = x.shape

        # embedding: always (seq_len, bs, embed_dim)
        x = self.embedding(x)

        # model may return (seq_len, bs, hidden) or (bs, seq_len, hidden)
        x, h = self.model(x, h)

        if x.shape == (bs, seq_len, self.hidden_size):
            # transformer-style -> swap
            x = x.transpose((1, 0, 2))

        # now safe to reshape
        x = self.linear(x.reshape((seq_len * bs, self.embedding_size)))

        return x, h



if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
