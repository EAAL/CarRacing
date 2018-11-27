import gym
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Reshape, Conv2D
from keras.optimizers import Adam

import keras.backend as K
import tensorflow as tf

class DQNAgent():
	def __init__(self, env, sess):
		self.env = env
		self.sess = sess
		self.epsilon = 1.0
		self.epsilon_decay = 0.999
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
		

def main():
	env = gym.make("CarRacing-v0")
	epoch = 1000
	steps = 900
	with tf.Session() as sess:
		agent = DQNAgent(env, sess)
		for ep in range(epoch):
			state = env.reset()
			done = False
			total_reward = 0.0
			for step in range(steps):
				env.render()
				action = agent.act(state)
				new_state, reward, done, info = env.step(action)
				total_reward += reward
				if done:
					break
			print(f'Episode {ep} done with reward {total_reward}')

if __name__ == "__main__":
	main()
