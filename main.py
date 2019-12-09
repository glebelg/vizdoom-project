import argparse
import numpy as np

from vizdoom import DoomGame

from utils import game_state
from trainer import Trainer

D1_CFG_PATH = "./scenarios/health_gathering.cfg"

def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.init()
    return game


def d1_mode(args):
    game = initialize_vizdoom(D1_CFG_PATH)

    trainer = Trainer(game, args)

    img = game_state(game)
    state = img.expand(4, -1, -1)

    loss_log, reward_log, health_log, timeout_log, reward_log_test, health_log_test, timeout_log_test = trainer.train(state)

    game.close()

    print(np.array(health_log_test).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-mode", default="D1", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=int)    
    parser.add_argument("--steps", default=525, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--test-episods", default=2, type=int)
    args = parser.parse_args()

    if args.game_mode == "D1":
        d1_mode(args)


if __name__ == "__main__":
    main()