import numpy as np


def batch_generator(train_x: np.ndarray, train_y: np.ndarray, batch_size: int):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    size = train_x.shape[0]
    index = 0
    while index < size:
        batch_x = train_x[index: index+batch_size]
        batch_y = train_y[index: index+batch_size]
        index += batch_size

        yield batch_x, batch_y