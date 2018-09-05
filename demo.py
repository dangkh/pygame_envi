import gym
import gym_dang
import pygame
from utilise import *


def main():
	env = gym.make('dang-v1')
	config_value = getValueConfig()
	env.config(config_value)
	print('Running gym example')
	for i_episode in range(3):
		print('Starting episode %d' % i_episode)
		observation = env.reset()
		print(observation)
		num_steps = 0
		while num_steps < 200:
			actions = env.get_action_list()
			print(actions)
			try:
				mode = input('Input:')
				mode = int(mode)
				mode = len(actions)-1 if mode >= len(actions) else mode
			except ValueError:
				print("Not a number")
				mode = 0
			action = actions[mode]
			print(num_steps, " ", actions[mode])
			observation, reward, done = env.step(mode)
			print reward
			if reward > 1: 
				print "done"
				break
			num_steps += 1



if __name__ == "__main__":
    main()
