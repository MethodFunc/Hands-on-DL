import gym
# import pyvirtualdisplay

env = gym.make("CartPole-v1")
obs = env.reset()
# print(obs)

# try:
#     import pyvirtualdisplay
#     display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
# except ImportError:
#     pass

env.render()