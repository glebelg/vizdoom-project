import argparse
import numpy as np

from vizdoom import DoomGame

from trainer import Trainer
from utils import game_state
from graphsDrawer import GraphsDrawer


D1_CFG_PATH = "./scenarios/health_gathering.cfg"
D2_CFG_PATH = "./scenarios/health_gathering_supreme.cfg"
D3_CFG_PATH = "./scenarios/battle.cfg"


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    game.set_window_visible(False)
    game.init()
    return game


def d1_mode(args):
    game = initialize_vizdoom(D1_CFG_PATH)
    state = game_state(game).expand(4, -1, -1)

    goal = [0, 1, 0]
    health = game.get_state().game_variables[0]
    measurement = [0, health, 0]

    trainer = Trainer(game, goal, measurement, args)
    trainer.train(state)

    game.close()

    # trainer.watch_test_episodes()


def d2_mode(args):
    game = initialize_vizdoom(D2_CFG_PATH)
    state = game_state(game).expand(4, -1, -1)

    goal = [0, 1, 0]
    health = game.get_state().game_variables[0]
    measurement = [0, health, 0]

    trainer = Trainer(game, goal, measurement, args)
    trainer.train(state)

    game.close()


def d3_mode(args):
    game = initialize_vizdoom(D3_CFG_PATH)
    state = game_state(game).expand(4, -1, -1)

    goal = [0.5, 0.5, 1]
    ammo = game.get_state().game_variables[0]
    health = game.get_state().game_variables[1]
    frag = game.get_state().game_variables[2]
    measurement = [ammo, health, frag]

    trainer = Trainer(game, goal, measurement, args)
    trainer.train(state)

    game.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-mode", default="D1", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--iterations", default=8000, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--test-episodes", default=80, type=int)
    args = parser.parse_args()


    if args.game_mode == "D1":
        d1_mode(args)
    elif args.game_mode == "D2":
        d2_mode(args)
    elif args.game_mode == "D3":
        d3_mode(args)
    else:
        raise Exception("game-mode must be D1, D2 or D3")


    graphs_drawer = GraphsDrawer(args)

    graphs_drawer.draw_loss()
    graphs_drawer.draw_train_total_reward()
    graphs_drawer.draw_test_total_reward()
    graphs_drawer.draw_finished_episodes()
    graphs_drawer.draw_episode_loss()
    graphs_drawer.draw_train_average_health()
    graphs_drawer.draw_test_average_health()
    if args.game_mode == "D3":
        graphs_drawer.draw_train_frags()
        graphs_drawer.draw_train_average_frags()
        graphs_drawer.draw_test_frags()


if __name__ == "__main__":
    main()
