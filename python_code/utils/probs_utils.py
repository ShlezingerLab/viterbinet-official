import numpy as np
import torch

from python_code import DEVICE


def calculate_siso_states(memory_length: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    calculates siso states vector for the transmitted words. Number of states is 2 ** memory length.
    Only BPSK is allowed.
    :param memory_length: length of channel memory
    :param transmitted_words: channel transmitted words
    :return: vector of length of transmitted_words with values in the range of 0,1,...,n_states-1
    """
    states_enumerator = (2 ** torch.arange(memory_length)).reshape(1, -1).float().to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def break_transmitted_siso_word_to_symbols(memory_length: int, transmitted_words: np.ndarray) -> np.ndarray:
    """
    Take words of bits b_0,..b_(n-1) with length n, and creates [memory_length X n-memory_length] matrix
    with bits b_0,..., b_(n-1-memory_length) at first row, bits b_1,..., b_(n-1-memory_length+1) at second row
    up to memory_length such rows
    """
    padded = np.concatenate([np.zeros([transmitted_words.shape[0], memory_length - 1]), transmitted_words,
                             np.zeros([transmitted_words.shape[0], memory_length])], axis=1)
    unsqueezed_padded = np.expand_dims(padded, axis=1)
    blockwise_words = np.concatenate([unsqueezed_padded[:, :, i:-memory_length + i] for i in range(memory_length)],
                                     axis=1)
    return blockwise_words.squeeze().T
