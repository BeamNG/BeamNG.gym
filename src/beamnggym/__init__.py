import os

from gymnasium.envs.registration import register

from . import envs

register(
    id='BNG-WCA-Race-Geometry-v0',
    entry_point='beamnggym.envs:WCARaceGeometry',
)

def read(fil):
    fil = os.path.join(os.path.dirname(__file__), fil)
    with open(fil, encoding='utf-8') as f:
        return f.read()

__version__ = read('version.txt')
