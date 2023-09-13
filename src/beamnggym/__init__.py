from gymnasium.envs.registration import register

from . import envs

register(
    id='BNG-WCA-Race-Geometry-v0',
    entry_point='beamnggym.envs:WCARaceGeometry',
)
