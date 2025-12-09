import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.p = p
        self.transforms = transforms
        
        # Load the appropriate data files
        if train:
            # Load training batches (data_batch_1 through data_batch_5)
            X_list = []
            y_list = []
            
            for i in range(1, 6):
                batch_file = os.path.join(base_folder, f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    X_list.append(batch_dict[b'data'])
                    y_list.append(batch_dict[b'labels'])
            
            # Concatenate all training batches
            self.X = np.concatenate(X_list, axis=0)
            self.y = np.concatenate(y_list, axis=0)
        else:
            # Load test batch
            test_file = os.path.join(base_folder, 'test_batch')
            with open(test_file, 'rb') as f:
                test_dict = pickle.load(f, encoding='bytes')
                self.X = test_dict[b'data']
                self.y = np.array(test_dict[b'labels'])
        
        # Reshape X from (N, 3072) to (N, 3, 32, 32)
        # CIFAR-10 images are 32x32x3, stored as flattened arrays of 3072 elements
        # The data is stored in row-major order: first 1024 entries are red channel,
        # next 1024 are green, last 1024 are blue
        self.X = self.X.reshape(-1, 3, 32, 32)
        
        # Normalize pixel values to 0-1 range
        self.X = self.X.astype(np.float32) / 255.0
        
        # Convert labels to numpy array if not already
        self.y = np.array(self.y, dtype=np.int64)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image = self.X[index]
        label = self.y[index]
        
        # Apply transforms if provided
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION