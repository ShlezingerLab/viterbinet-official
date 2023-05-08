import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.utils.constants import HALF

GAMMA = 0.5  # gamma value for time decay SISO fading
FADING_TAPS = [51, 39, 33, 21]
MAX_TAP = 0.8
VAR_TAP = 0.2
TEN = 10


class ISIAWGNChannel:
    @staticmethod
    def calculate_channel(memory_length: int, fading: bool = False, index: int = 0) -> np.ndarray:
        h = np.reshape(np.exp(-GAMMA * np.arange(memory_length)), [1, memory_length])
        h = ISIAWGNChannel._add_fading(h, memory_length, index) if fading else MAX_TAP * h
        return h

    @staticmethod
    def _add_fading(h: np.ndarray, memory_length: int, index: int) -> np.ndarray:
        h *= (MAX_TAP + VAR_TAP * np.cos(2 * np.pi * index / np.array(FADING_TAPS))).reshape(1, memory_length)
        return h

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, memory_length: int, s: np.ndarray) -> np.ndarray:
        blockwise_s = np.concatenate([s[:, i:-memory_length + i] for i in range(memory_length)], axis=0)
        conv = np.dot(h[:, ::-1], blockwise_s)
        return conv

    @staticmethod
    def _sample_noise_vector(row: int, col: int, snr: float) -> np.ndarray:
        noise_generator = default_rng(seed=conf.seed)
        snr_value = TEN ** (snr / TEN)
        w = (snr_value ** (-HALF)) * noise_generator.standard_normal((row, col))
        return w

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, memory_length: int) -> np.ndarray:
        conv = ISIAWGNChannel._compute_channel_signal_convolution(h, memory_length, s)
        [row, col] = conv.shape
        w = ISIAWGNChannel._sample_noise_vector(row, col, snr)
        y = conv + w
        return y
