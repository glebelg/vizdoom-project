import os
import numpy as np
import matplotlib.pyplot as plt


class GraphsDrawer:
    def __init__(self, args):
        self.args = args
        self.logs_path = "./logs/game-mode_{}/".format(args.game_mode)

        if not os.path.isdir("./graphs/"):
            os.mkdir("./graphs/")


    def draw_loss(self):
        f = open(self.logs_path + "train_loss.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Train loss per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig("./graphs/{}-train_loss.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_train_total_reward(self):
        f = open(self.logs_path + "train_total_reward.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Total reward per train episode")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.savefig("./graphs/{}-train_total_reward.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_test_total_reward(self):
        f = open(self.logs_path + "test_total_reward.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Average total reward per test episodes after epoch")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.savefig("./graphs/{}-test_total_reward.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_finished_episodes(self):
        f = open(self.logs_path + "train_episodes_finished.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Number of finished episodes per train eposh")
        plt.xlabel("Epoch")
        plt.ylabel("Finished episode")
        plt.savefig("./graphs/{}-train_episodes_finished.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_episode_loss(self):
        f = open(self.logs_path + "train_episode_loss.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Train loss per episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.savefig("./graphs/{}-train_episode_loss.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_train_average_health(self):
        f = open(self.logs_path + "train_average_health.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Average health per train episode")
        plt.xlabel("Episode")
        plt.ylabel("Average health")
        plt.savefig("./graphs/{}-train_average_health.png".format(self.args.game_mode))

        plt.close()
        f.close()

    
    def draw_test_average_health(self):
        f = open(self.logs_path + "test_average_health.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Grand mean health per test episodes after epoch")
        plt.xlabel("Episode")
        plt.ylabel("Average health")
        plt.savefig("./graphs/{}-test_average_health.png".format(self.args.game_mode))

        plt.close()
        f.close()
    

    def draw_train_frags(self):
        f = open(self.logs_path + "train_frags.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Number of frags per train episode")
        plt.xlabel("Episode")
        plt.ylabel("Frags")
        plt.savefig("./graphs/{}-train_frags.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_train_average_frags(self):
        f = open(self.logs_path + "train_average_frags.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Average number of frags per train epoch")
        plt.xlabel("Episode")
        plt.ylabel("Frags")
        plt.savefig("./graphs/{}-train_average_frags.png".format(self.args.game_mode))

        plt.close()
        f.close()


    def draw_test_frags(self):
        f = open(self.logs_path + "test_frags.log")
        data = np.array(f.read().splitlines()).astype(float)

        plt.plot(range(len(data)), data)
        plt.title("Average number of frags per test episodes after epoch")
        plt.xlabel("Episode")
        plt.ylabel("Frags")
        plt.savefig("./graphs/{}-test_frags.png".format(self.args.game_mode))

        plt.close()
        f.close()
        