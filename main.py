import argparse
import numpy as np

from vizdoom import DoomGame

from utils import game_state
from trainer import Trainer


D1_CFG_PATH = "./scenarios/health_gathering.cfg"
D2_CFG_PATH = "./scenarios/..."
D3_CFG_PATH = "./scenarios/..."


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.init()
    return game


def d1_mode(args):
    game = initialize_vizdoom(D1_CFG_PATH)

    goal = [0, 1, 0]
    health = game.get_state().game_variables[0]
    measurement = [0, health, 0]

    trainer = Trainer(game, goal, measurement, args)

    img = game_state(game)
    state = img.expand(4, -1, -1)

    loss_log, reward_log, health_log, timeout_log, reward_log_test, health_log_test, timeout_log_test = trainer.train(state)

    game.close()

    print(np.array(health_log_test).mean())


def d2_mode(args):
    pass


def d3_mode(args):
    pass


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
    elif args.game_mode == "D2":
        d2_mode(args)
    elif args.game_mode == "D3":
        d3_mode(args)


if __name__ == "__main__":
    main()