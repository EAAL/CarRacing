import gym
import numpy as np

from collections import deque

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Reshape, Conv2D
from keras.optimizers import Adam

import keras.backend as K
import tensorflow as tf

class DQNAgent():
	def __init__(self, env):
		self.env = env
		self.epsilon = 1.0
		self.epsilon_decay = 0.999
		self.batch_size = 100
		self.memory = deque(maxlen=10000)
		self.model = self.create_model()

	def create_model(self):
		i = Input(shape=(96, 96, 3, ))
		fl = Flatten()(i)
		l1 = Dense(30)(fl)
		l2 = Dense(30)(l1)
		o = Dense(2)(l2)
		model = Model(i, o) 
		model.compile(optimizer='rmsprop', loss='mse')
		return model

	def random_action(self):
		steer = np.random.random()*2.0-1.0
		acc = np.random()*2.0-1.0
		return np.array([steer, max(acc, 0.0), min(acc, 0.0)])

	def act(self, state):
		self.epsilon *= self.epsilon_deacy
		if self.epsilon < np.random.random():
			return self.random_action()
		action = self.model.predict(state)
		return np.array([action[0], max(action[1], 0.0), min(action[1], 0.0)])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train(self):
        samples = np.random.choice(self.memory, self.batch_size)
        self.model.fit()
		

def main():
	env = gym.make("CarRacing-v0")
	epoch = 1000
	steps = 900
	agent = DQNAgent(env)
	for ep in range(epoch):
		state = env.reset()
		done = False
		total_reward = 0.0
		for step in range(steps):
			env.render()
			action = agent.act(state)
			new_state, reward, done, info = env.step(action)
			agent.remember(state, action, reward, new_state, done)
			total_reward += reward
			state = new_state
			if done:
				break
		print(f'Episode {ep} done with reward {total_reward}')
		agent.train()

if __name__ == "__main__":
	main()
