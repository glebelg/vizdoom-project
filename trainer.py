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
from utils import game_state, save_data, save_model


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

        self.model = Model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr)

        self.memory = ReplayMemory()


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
        measurement_input = torch.zeros(batch_size, 3).to(self.device)
        goal_input = self.goal.expand(batch_size, -1).to(self.device)
        f_action_target = torch.zeros(batch_size, 3 * 6).to(self.device)
        
        action = []
        self.memory_batch = self.memory.get_sample(self.memory.size)
        
        for i, idx in enumerate(rand_indices):
            future_measurements = []
            last_offset = 0
            done = False
            for j in range(self.timesteps[-1] + 1):
                if not self.memory_batch[4][idx + j]: # if episode is not finished
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
            
            ammo = self.measurement[0]
            health = self.measurement[1]
            frag = self.measurement[2]

        else:
            isterminal = 0.
            img2 = game_state(self.game).to(self.device)
            s2 = torch.cat([s1[:3], img2])
        
            ammo = self.game.get_state().game_variables[0]
            health = self.game.get_state().game_variables[1]
            frag = self.game.get_state().game_variables[2]
            
        measurement = np.divide([ammo, health, frag], [7.5, 30., 1.]).astype(np.float32)
        self.measurement = torch.from_numpy(measurement).to(self.device)   
        self.memory.add_transition(s1, a, s2, self.measurement, isterminal, reward)
            
        loss = self.train_minibatch()
        
        return loss


    def test(self):
        scores = np.array([])
        loss_log, reward_log, health_log, timeout_log = [], [], [], []

        
        img = game_state(self.game)
        state = img.expand(4, -1, -1).to(self.device)
        
        for _ in trange(self.args.test_episodes):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                h = self.game.get_state().game_variables[0]
                img = game_state(self.game).to(self.device)
                state = torch.cat([state[:3], img])
                
                a_idx = self.get_best_action(state)
                
                self.game.make_action(self.actions[a_idx], 12)
                
            r = self.game.get_total_reward()
            reward_log.append(r) # make file
            scores = np.append(scores, r) # make file
            timeout_log.append(self.game.get_episode_timeout()) # make file
            
            health_log.append(h)
                
        return reward_log, health_log, timeout_log


    def train(self, state):
        loss_log, total_reward_log, health_log, timeout_log = [], [], [], []
        reward_log_test, health_log_test, timeout_log_test = [], [], []
        
        for epoch in range(self.args.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.args.epochs))
            print("Training...")

            episodes_finished = 0
            scores = np.array([])
            self.game.new_episode()
            
            for learning_step in trange(self.args.steps):
                loss = self.perform_learning_step(epoch, state)
                
                health_log.append(self.measurement) # make file

                if loss is not None:
                    loss_log.append(loss) # make file 
                
                if self.game.is_episode_finished():
                    total_reward_log.append(self.game.get_total_reward())
                    
                    self.game.new_episode()
                    episodes_finished += 1
                    
                    timeout_log.append(self.game.get_episode_timeout())

            save_data(total_reward_log, "train_total_reward", self.args)

                    
            print("Completed {} episodes".format(episodes_finished))
            print("Testing...")
            reward_log_test_, health_log_test_, timeout_log_test_ = self.test()
            
            reward_log_test.append(reward_log_test_)
            health_log_test.append(health_log_test_)
            timeout_log_test.append(timeout_log_test_)
            
            # save_model(self.model)


    def watch_test_episodes(self):
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()
        for episode in range(1):
            self.game.new_episode("episode-%d" % episode)

            img = game_state(self.game).to(self.device)
            state = img.expand(4, -1, -1)

            while not self.game.is_episode_finished():
                img = game_state(self.game).to(self.device)
                state = torch.cat([state[:3], img]).to(self.device)

                # state = game_state(self.game)
                # state = state.reshape([1, 1, 84, 84])

                a_idx = self.get_best_action(state)
                self.game.set_action(self.actions[a_idx])

                for _ in range(12):
                    self.game.advance_action()

            sleep(0.02)
            score = self.game.get_total_reward()
            print("Total score:", score)