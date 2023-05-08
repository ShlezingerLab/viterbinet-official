import torch

from python_code.detectors.trainer import Trainer
from python_code.detectors.viterbinet.viterbinet_detector import ViterbiNetDetector
from python_code.utils.constants import Phase, MEMORY_LENGTH
from python_code.utils.probs_utils import calculate_siso_states

EPOCHS = 500


class ViterbiNetTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        self.n_states = 2 ** MEMORY_LENGTH
        self.is_online_training = True
        super().__init__()

    def __str__(self):
        return 'ViterbiNet'

    def _initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = ViterbiNetDetector(n_states=self.n_states)

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est: [1,transmission_length,n_states], each element is a probability
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_siso_states(MEMORY_LENGTH, tx)
        loss = self.criterion(input=est, target=gt_states)
        return loss

    def _forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase=Phase.TEST)
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous calculated weights.
        :param tx: transmitted word
        :param rx: received word
        :param h: channel coefficients
        """
        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float(), phase=Phase.TRAIN)
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx)
            loss += current_loss
