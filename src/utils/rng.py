import random

import numpy as np
import torch


def seed(seed: int = 0):
    """Set the random seed of modules :module:`random`, :module:`numpy` and :module:`torch`"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state():
    """Get the random state of modules :module:`random`, :module:`numpy` and :module:`torch`"""
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()
    return random_state, np_random_state, torch_random_state


def set_rng_state(random_state, np_random_state, torch_random_state):
    """Set the random state of modules :module:`random`, :module:`numpy` and :module:`torch`"""
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    torch.set_rng_state(torch_random_state)
