import random
import torch
import numpy as np

def set_seeds(
    seed: int
    ) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def r2_acc_to_loss(
    r2_acc: float
    )-> float:
    return (-1 * r2_acc) + 1