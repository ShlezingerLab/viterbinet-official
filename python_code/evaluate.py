import os

from python_code.detectors.viterbinet.viterbinet_trainer import ViterbiNetTrainer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    trainer = ViterbiNetTrainer()
    print(trainer)
    trainer.evaluate()
