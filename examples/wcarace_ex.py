from random import uniform

import gymnasium as gym
import beamnggym  # noqa: F401

SIMULATOR_HOME_DIR = '/path/to/your/BeamNG_Tech_folder'

env = gym.make('BNG-WCA-Race-Geometry-v0', home=SIMULATOR_HOME_DIR)
while True:
    print('Resetting environment...')
    env.reset()
    total_reward, terminated, truncated = 0, False, False
    # Drive around randomly until finishing
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step((uniform(-1, 1), uniform(-1, 1) * 10))
        total_reward += reward
    print('Achieved reward:', total_reward)
