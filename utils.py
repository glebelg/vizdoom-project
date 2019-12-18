import torch

import os
import numpy as np

from skimage.transform import resize


def preprocess(img):
    return torch.from_numpy(resize(img, (1, 84, 84)).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)

    
def save_model(model):
    model_path = "./saved_models/model-doom_{}.pth".format(self.args.game_mode)
    torch.save(model, model_path)
