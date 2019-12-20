import torch
import itertools
import numpy as np
from time import sleep

from vizdoom import DoomGame, Mode

from utils import game_state
from main import initialize_vizdoom


D2_CFG_PATH = "./scenarios/health_gathering_supreme.cfg"

device = torch.device("cpu")


def get_best_action(state):
    out = model(state.unsqueeze(0), measurement.unsqueeze(0), goal.unsqueeze(0))
    pred = goal @ out.view(3 * 6, -1)
    index = torch.argmax(pred)
    return index


def watch_test_episodes(game):
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for episode in range(1):
        game.new_episode("episode-D2")

        img = game_state(game).to(device)
        state = img.expand(4, -1, -1)

        while not game.is_episode_finished():
            img = game_state(game).to(device)
            state = torch.cat([state[:3], img]).to(device)

            a_idx = get_best_action(state)
            game.set_action(actions[a_idx])

            for _ in range(12):
                game.advance_action()

        sleep(0.02)
        score = game.get_total_reward()
        print("Total score:", score)


game = initialize_vizdoom(D2_CFG_PATH)

action_size = game.get_available_buttons_size()
actions = [list(a) for a in itertools.product([0, 1], repeat=action_size)]

measurement_size = 3 # [ ammo count, health level, frag count ]
timesteps = [1, 2, 4, 8, 16, 32]

goal = [0, 1, 0]
goal = torch.FloatTensor(goal * len(timesteps)).to(device)

health = game.get_state().game_variables[0]
measurement = [0, health, 0]
measurement = np.divide(measurement, [7.5, 30., 1.]).astype(np.float32)
measurement = torch.FloatTensor(torch.from_numpy(measurement).to(device))

model = torch.load("./saved_models/model-doom_D1.pth",  map_location=torch.device('cpu'))

game.close()

watch_test_episodes(game)
