import torch

import os
import numpy as np

from skimage.transform import resize


def preprocess(img):
    return torch.from_numpy(resize(img, (1, 84, 84)).astype(np.float32))


def game_state(game):
    return preprocess(game.get_state().screen_buffer)


def save_data(data, file, args):
    if not os.path.isdir("./logs/"):
        os.mkdir("./logs/")

    folder = "./logs/game-mode_{}/".format(args.game_mode)
    name_args = [file, args.batch_size, args.epochs, args.steps, args.test_episodes]
    file = "{}-batchSize_{}-epochs_{}-steps-{}-testEpisodes_{}.log".format(*name_args)

    if not os.path.isdir(folder):
        os.mkdir(folder)

    with open(folder + file, 'w') as f:
        for item in data:
            f.write("{}\n".format(str(item)))

    
def save_model(model):
    model_path = "./saved_models/model-doom_{}.pth".format(self.args.game_mode)
    torch.save(model, model_path)
