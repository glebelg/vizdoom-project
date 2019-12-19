import torch
import torch.nn as nn

import itertools
import numpy as np

from time import sleep
from vizdoom import Mode
from tqdm import tqdm, trange
from random import random, randint

from model import Model
from memory import ReplayMemory
from logsKeeper import LogsKeeper
from utils import game_state, save_model


STOP_EPISODE = 525

class Trainer:
    def __init__(self, game, goal, measurement, args):
        self.game = game
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        action_size = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in itertools.product([0, 1], repeat=action_size)]

        measurement_size = 3 # [ ammo count, health level, frag count ]
        self.timesteps = [1, 2, 4, 8, 16, 32]

        self.goal = torch.FloatTensor(goal * len(self.timesteps)).to(self.device)

        measurement = np.divide(measurement, [7.5, 30., 1.])
        self.measurement = torch.from_numpy(measurement).to(self.device)

        self.model = Model(len(self.actions)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-4)

        self.memory = ReplayMemory()
        self.logsKeeper = LogsKeeper(self.args)


    def get_best_action(self, state):
        out = self.model(state.unsqueeze(0), self.measurement.unsqueeze(0), self.goal.unsqueeze(0))
        pred = self.goal @ out.view(3 * 6, -1)    
        index = torch.argmax(pred)
        return index


    def train_minibatch(self):
        if self.memory.size <= (self.timesteps[-1] + 1):
            return None
        
        batch_size = min(64, self.memory.size)
        rand_indices = np.random.choice(self.memory.size - (self.timesteps[-1] + 1), batch_size)

        state_input = torch.zeros(batch_size, 4, 84, 84).to(self.device)
        measurement_input = torch.zeros(batch_size, len(self.measurement)).to(self.device)
        goal_input = self.goal.expand(batch_size, -1).to(self.device)
        f_action_target = torch.zeros(batch_size, len(self.measurement) * len(self.timesteps)).to(self.device)
        
        action = []
        self.memory_batch = self.memory.get_sample(self.memory.size)
        
        for i, idx in enumerate(rand_indices):
            future_measurements = []
            last_offset = 0
            done = False
            for j in range(self.timesteps[-1] + 1):
                if not self.memory_batch[4][idx + j]:
                    if j in self.timesteps:
                        if not done:
                            future_measurements += list((self.memory_batch[3][idx + j] - self.memory_batch[3][idx]))
                            last_offset = j
                        else:
                            future_measurements += list((self.memory_batch[3][idx + last_offset] - self.memory_batch[3][idx]))
                else:
                    done = True
                    if j in self.timesteps:
                        future_measurements += list((self.memory_batch[3][idx + last_offset] - self.memory_batch[3][idx]))

            f_action_target[i,:] = torch.tensor(future_measurements)
            state_input[i,:,:,:] = self.memory_batch[0][idx]
            measurement_input[i,:] = self.memory_batch[3][idx]
            action.append(self.memory_batch[1][idx])

        f_target = self.model(state_input, measurement_input, goal_input)
        f_target = f_target.view(-1, 3 * 6)
        
        for i in range(batch_size):
            f_target[action[i].long()] = f_action_target[i]

        pred = self.model(state_input, measurement_input, goal_input)
        pred = pred.view(-1, 3 * 6)
        
        loss = self.criterion(pred, f_target)
        self.optimizer.zero_grad()
        loss.backward()

        if nn.utils.clip_grad_norm_(self.model.parameters(), 0.25) < 4:
            self.optimizer.step()

        return loss.item()


    def find_eps(self, epoch):
        start, end = 1.0, 0.1
        const_epochs, decay_epochs = 1., 6.
        if epoch < const_epochs:
            return start
        elif epoch > decay_epochs:
            return end

        progress = (epoch - const_epochs) / (decay_epochs - const_epochs)
        return start - progress * (start - end)


    def perform_learning_step(self, epoch, state):
        img1 = game_state(self.game)
        s1 = torch.cat([state[:3], img1]).to(self.device)
        
        if random() <= self.find_eps(epoch):
            a = torch.tensor(randint(0, len(self.actions) - 1)).long()
        else:
            a = self.get_best_action(s1)
            
        reward = self.game.make_action(self.actions[a], 12)
        
        if self.game.is_episode_finished():
            isterminal, s2 = 1., None
            
            if self.args.game_mode == "D3":
                ammo = self.measurement[0]
                health = self.measurement[1]
                frag = self.measurement[2]
            else:
                ammo = 0
                health = self.measurement[0]
                frag = 0

        else:
            isterminal = 0.
            img2 = game_state(self.game).to(self.device)
            s2 = torch.cat([s1[:3], img2])

            if self.args.game_mode == "D3":
                ammo = self.game.get_state().game_variables[0]
                health = self.game.get_state().game_variables[1]
                frag = self.game.get_state().game_variables[2]
            else:
                ammo = 0
                health = self.game.get_state().game_variables[0]
                frag = 0
            
        measurement = np.divide([ammo, health, frag], [7.5, 30., 1.]).astype(np.float32)
        self.measurement = torch.from_numpy(measurement).to(self.device)   
        self.memory.add_transition(s1, a, s2, self.measurement, isterminal, reward)
            
        loss = self.train_minibatch()
        
        return loss


    def test(self):
        health, total_reward, frags = [], [], []

        img = game_state(self.game)
        state = img.expand(4, -1, -1).to(self.device)
        
        for _ in trange(self.args.test_episodes):
            self.game.new_episode()

            cur_health = []

            while not self.game.is_episode_finished():
                cur_health.append(self.game.get_state().game_variables[int(self.args.game_mode == "D3")])
                if self.args.game_mode == "D3":
                    frag = self.game.get_state().game_variables[2]

                img = game_state(self.game).to(self.device)
                state = torch.cat([state[:3], img])
                
                a_idx = self.get_best_action(state)
                self.game.make_action(self.actions[a_idx], 12)

            health.append(np.array(cur_health).mean())
            total_reward.append(self.game.get_total_reward())
            if self.args.game_mode == "D3":
                frags.append(frag)

        self.logsKeeper.save_val(np.array(total_reward).mean(), "test_total_reward.log")
        self.logsKeeper.save_val(np.array(health).mean(), "test_average_health.log")
        if self.args.game_mode == "D3":
            self.logsKeeper.save_val(np.array(frags).mean(), "test_frags.log")


    def train(self, state):
        for epoch in range(self.args.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.args.epochs))
            print("Training...")

            episodes_finished = 0
            self.game.new_episode()
            cut_step = 0

            total_reward = []
            health = [self.game.get_state().game_variables[int(self.args.game_mode == "D3")]]
            if self.args.game_mode == "D3":
                frags = [self.game.get_state().game_variables[2]]
    
            for learning_step in trange(self.args.iterations):
                loss = self.perform_learning_step(epoch, state)
                cut_step += 1

                health.append(self.measurement[1].item() * 30)

                self.logsKeeper.save_measurement(self.measurement, "train_measurement.log")
                if loss is not None:
                    self.logsKeeper.save_val(loss, "train_loss.log")
                
                if self.game.is_episode_finished() or cut_step == STOP_EPISODE:
                    self.logsKeeper.save_val(self.game.get_total_reward(), "train_total_reward.log")
                    self.logsKeeper.save_val(self.measurement[2].item(), "train_frags.log")
                    self.logsKeeper.save_val(np.array(health).mean(), "train_average_health.log")
                    if loss is not None:
                        self.logsKeeper.save_val(loss, "train_episode_loss.log")

                    if self.args.game_mode == "D3":
                        frags.append(self.measurement[2].item())

                    total_reward.append(self.game.get_total_reward())

                    self.game.new_episode()
                    episodes_finished += 1
                    health = []
                    cut_step = 0
                    
            self.logsKeeper.save_val(np.array(total_reward).mean(), "train_average_total_reward.log")
            self.logsKeeper.save_val(episodes_finished, "train_episodes_finished.log")
            if self.args.game_mode == "D3":
                self.logsKeeper.save_val(np.array(frags).mean(), "train_average_frags.log")

            print("Completed {} episodes".format(episodes_finished))
            print("Testing...")
            self.test()
            
            save_model(self.model, self.args)


    def watch_test_episodes(self):
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()
        for episode in range(1):
            self.game.new_episode("episode-%d" % episode)

            img = game_state(self.game).to(self.device)
            state = img.expand(4, -1, -1)

            while not self.game.is_episode_finished():
                img = game_state(self.game).to(self.device)
                state = torch.cat([state[:3], img]).to(self.device)

                a_idx = self.get_best_action(state)
                self.game.set_action(self.actions[a_idx])

                for _ in range(12):
                    self.game.advance_action()

            sleep(0.02)
            score = self.game.get_total_reward()
            print("Total score:", score)