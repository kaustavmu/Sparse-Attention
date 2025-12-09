"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # Read images file
    with gzip.open(image_filename, 'rb') as f:
        # Read header: magic number, number of images, rows, cols
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number for images: {magic}"
        
        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, rows * cols)
        X = image_data.reshape(num_images, rows * cols).astype(np.float32)
        # Normalize to [0, 1]
        X = X / 255.0
    
    # Read labels file
    with gzip.open(label_filename, 'rb') as f:
        # Read header: magic number, number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number for labels: {magic}"
        assert num_images == num_labels, "Mismatch between number of images and labels"
        
        # Read label data
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    
    # Compute log softmax: log(exp(z_i) / sum(exp(z_j)))
    # = z_i - log(sum(exp(z_j)))
    log_sum_exp = ndl.ops.logsumexp(Z, axes=(1,))  # Shape: (batch_size,)
    log_sum_exp = ndl.ops.reshape(log_sum_exp, (batch_size, 1))  # Shape: (batch_size, 1)
    
    log_softmax = Z - log_sum_exp  # Shape: (batch_size, num_classes)
    
    # Compute cross entropy loss: -sum(y * log_softmax)
    loss_per_sample = -ndl.ops.summation(y_one_hot * log_softmax, axes=(1,))  # Shape: (batch_size,)
    
    # Return average loss
    avg_loss = ndl.ops.summation(loss_per_sample) / batch_size
    
    return avg_loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    
    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        
        # Get batch data
        X_batch = ndl.Tensor(X[start:end])
        y_batch = y[start:end]
        
        # Create one-hot encoding for labels
        y_one_hot = np.zeros((end - start, num_classes))
        y_one_hot[np.arange(end - start), y_batch] = 1
        y_one_hot_tensor = ndl.Tensor(y_one_hot)
        
        # Forward pass: logits = ReLU(X * W1) * W2
        Z1 = ndl.ops.matmul(X_batch, W1)  # (batch_size, hidden_dim)
        A1 = ndl.ops.relu(Z1)  # (batch_size, hidden_dim)
        Z2 = ndl.ops.matmul(A1, W2)  # (batch_size, num_classes)
        
        # Compute loss
        loss = softmax_loss(Z2, y_one_hot_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
    
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        X, y = batch
        
        if opt is not None:
            opt.reset_grad()
        
        # Forward pass
        logits = model(X)
        loss = loss_fn(logits, y)
        
        # Backward pass if training
        if opt is not None:
            loss.backward()
            opt.step()
        
        # Compute accuracy
        predictions = np.argmax(logits.numpy(), axis=1)
        correct = np.sum(predictions == y.numpy())
        
        # Accumulate statistics
        total_loss += loss.numpy() * X.shape[0]
        total_correct += correct
        total_samples += X.shape[0]
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Create optimizer instance
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn_instance = loss_fn()
    
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn_instance, opt)
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn_instance = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn_instance, opt=None)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    nbatch, batch_size = data.shape
    h = None  # Initial hidden state
    
    for i in range(0, nbatch - 1, seq_len):
        # Get sequence data
        seq_end = min(i + seq_len, nbatch - 1)
        actual_seq_len = seq_end - i
        
        X = ndl.Tensor(data[i:seq_end], device=device, dtype=dtype)  # (seq_len, batch_size)
        y = ndl.Tensor(data[i+1:seq_end+1], device=device, dtype=dtype)  # (seq_len, batch_size)
        
        if opt is not None:
            opt.reset_grad()
        
        # Forward pass
        logits, h = model(X, h)  # logits: (seq_len * batch_size, vocab_size)
        
        # Detach hidden state to prevent backprop through entire sequence
        if h is not None:
            if isinstance(h, tuple):  # LSTM case
                h = (ndl.Tensor(h[0].numpy(), device=device, dtype=dtype), 
                     ndl.Tensor(h[1].numpy(), device=device, dtype=dtype))
            else:  # RNN case
                h = ndl.Tensor(h.numpy(), device=device, dtype=dtype)
        
        # Reshape targets for loss computation
        y_flat = ndl.ops.reshape(y, (actual_seq_len * batch_size,))  # (seq_len * batch_size,)
        
        # Compute loss
        loss = loss_fn(logits, y_flat)
        
        # Backward pass if training
        if opt is not None:
            loss.backward()
            
            # Gradient clipping if specified
            if clip is not None:
                # Compute gradient norm
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += np.sum(param.grad.numpy() ** 2)
                grad_norm = np.sqrt(grad_norm)
                
                # Clip gradients if norm exceeds threshold
                if grad_norm > clip:
                    scale = clip / grad_norm
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad = ndl.Tensor(param.grad.numpy() * scale)
            
            opt.step()
        
        # Compute accuracy
        predictions = np.argmax(logits.numpy(), axis=1)
        correct = np.sum(predictions == y_flat.numpy())
        
        # Accumulate statistics
        total_loss += loss.numpy() * actual_seq_len * batch_size
        total_correct += correct
        total_samples += actual_seq_len * batch_size
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Create optimizer instance
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn_instance = loss_fn()
    
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn_instance, opt, clip, device, dtype)
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn_instance = loss_fn()
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn_instance, opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)