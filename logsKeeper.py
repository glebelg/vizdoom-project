import os
import numpy as np


class LogsKeeper:
    def __init__(self, args):
        self.folder = "./logs/game-mode_{}/".format(args.game_mode)

        if not os.path.isdir("./logs/"):
            os.mkdir("./logs/")

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self._clear_logs()


    def _clear_logs(self):
        for file in os.listdir(self.folder):
            with open(self.folder + file, 'w'):
                pass


    def save_val(self, val, file):
        with open(self.folder + file, 'a') as f:
            f.write("{}\n".format(str(val)))


    def save_measurement(self, measurement, file):
        measurement = np.array([measurement[0].item(), measurement[1].item(), measurement[2].item()])
        measurement *= np.array([7.5, 30., 1.])

        with open(self.folder + file, 'a') as f:
            f.write("{} {} {}\n".format(*measurement.astype("str")))
