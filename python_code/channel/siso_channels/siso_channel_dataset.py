from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.modulator import BPSKModulator
from python_code.channel.siso_channels.isi_awgn_channel import ISIAWGNChannel
from python_code.utils.constants import MEMORY_LENGTH
from python_code.utils.probs_utils import break_transmitted_siso_word_to_symbols



class SISOChannel:
    def __init__(self, block_length: int, pilots_length: int, fading_in_channel: bool):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = MEMORY_LENGTH
        self.rx_length = 1
        self._h_shape = [self.rx_length, self.tx_length]
        self._fading_in_channel = fading_in_channel
        self._channel_model = ISIAWGNChannel
        self._modulator = BPSKModulator

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # create pilots and data
        tx_pilots = self._bits_generator.integers(0, 2, size=(1, self._pilots_length)).reshape(1, -1)
        tx_data = self._bits_generator.integers(0, 2, size=(1, self._block_length - self._pilots_length))
        tx = np.concatenate([tx_pilots, tx_data], axis=1).reshape(1, -1)
        # add zero bits
        padded_tx = np.concatenate(
            [np.zeros([tx.shape[0], MEMORY_LENGTH - 1]), tx, np.zeros([tx.shape[0], MEMORY_LENGTH])], axis=1)
        # modulation
        s = self._modulator.modulate(padded_tx)
        # transmit through noisy channel
        rx = self._channel_model.transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        tx, rx = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, tx), rx.T
        # return the symbols received before the transmission's end (last memory_length bits)
        return tx[:-MEMORY_LENGTH + 1], rx[:-MEMORY_LENGTH + 1]

    def _transmit_and_detect(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = self._channel_model.calculate_channel(MEMORY_LENGTH, fading=self._fading_in_channel, index=index)
        tx, rx = self._transmit(h, snr)
        return tx, h, rx
