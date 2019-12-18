import os
import numpy as np


class LogsKeeper:
    def __init__(self, args):
        self.name_args = [args.batch_size, args.epochs, args.steps, args.test_episodes]
        self.folder = "./logs/game-mode_{}/".format(args.game_mode)

        if not os.path.isdir("./logs/"):
            os.mkdir("./logs/")

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self._clear_logs()

    def _clear_logs(self):
        log_files = ["train_total_reward", "train_episodes_finished", "train_measurement"]

        for file in log_files:
            file = "{}-batchSize_{}-epochs_{}-steps-{}-testEpisodes_{}.log".format(file, *self.name_args)

            with open(self.folder + file, 'w'):
                pass


    def save_val(self, val, file):
        file = "{}-batchSize_{}-epochs_{}-steps-{}-testEpisodes_{}.log".format(file, *self.name_args)

        with open(self.folder + file, 'a') as f:
            f.write("{}\n".format(str(val)))

    def save_measurement(self, measurement, file):
        file = "{}-batchSize_{}-epochs_{}-steps-{}-testEpisodes_{}.log".format(file, *self.name_args)

        measurement = np.array([measurement[0].item(), measurement[1].item(), measurement[2].item()])
        measurement *= np.array([7.5, 30., 1.])

        with open(self.folder + file, 'a') as f:
            f.write("{} {} {}\n".format(*measurement.astype("str")))
