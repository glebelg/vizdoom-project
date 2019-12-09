import torch

import numpy as np

from skimage.transform import resize


def preprocess(img):
    return torch.from_numpy(resize(img, (1, 84, 84)).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)