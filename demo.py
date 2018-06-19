import gym
import gym_dang
import pygame


def load_image(image, darken, colorKey = None):
    ret = []
    if colorKey is not None:
        image.set_colorkey(colorKey)
    if darken:
        image.set_alpha(127)
    for i in range(image.get_width()):
        s = pygame.Surface((1, image.get_height())).convert()
        s.blit(image, (- i, 0))
        if colorKey is not None:
            s.set_colorkey(colorKey)
        ret.append(s)
    return ret

def getValueConfig():
	size = w, h = 640,480
	pygame.init()
	pygame.display.set_mode(size)
	sprites = [  
	      load_image(pygame.image.load("gym_dang/envs/pics/items/barrel.png").convert(), False, colorKey = (0,0,0)),
	      load_image(pygame.image.load("gym_dang/envs/pics/items/pillar.png").convert(), False, colorKey = (0,0,0)),
	      load_image(pygame.image.load("gym_dang/envs/pics/items/greenlight.png").convert(), False, colorKey = (0,0,0)),
    ]
   	images = [  
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/eagle.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/redbrick.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/purplestone.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/greystone.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/bluestone.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/mossy.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/wood.png").convert(), False),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/colorstone.png").convert(), False),

	      load_image(pygame.image.load("gym_dang/envs/pics/walls/eagle.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/redbrick.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/purplestone.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/greystone.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/bluestone.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/mossy.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/wood.png").convert(), True),
	      load_image(pygame.image.load("gym_dang/envs/pics/walls/colorstone.png").convert(), True),
              ]
	background = pygame.transform.scale(pygame.image.load("gym_dang/envs/pics/background.png").convert(), (w,h))
	# pygame.quit()
	return [sprites, images, background]

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
		print("ok")
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
			num_steps += 1



if __name__ == "__main__":
    main()
