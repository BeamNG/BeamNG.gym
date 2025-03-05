# BeamNG.gym

BeamNG.gym is a collection of [Gymnasium](https://gymnasium.farama.org/)
environments that cover various driving tasks simulated in
[BeamNG.tech](https://beamng.tech/).

## Installation

Standard pip can be used to obtain the package of environments:

```bash
pip install beamng.gym
```

Or install the version from source by:

```bash
git clone https://github.com/BeamNG/BeamNG.gym.git
cd BeamNG.gym
pip install --editable .
```

A copy of [BeamNG.tech](https://beamng.tech/) is also required to
actually run the scenario. The basic version is freely available for academic non-commercial use.

This version is compatible with BeamNG.tech 0.30 and BeamNGpy 1.27.

## Configuration

The environments assume an envirionment variable to be set that specifies where
[BeamNG.tech](https://beamng.tech/) has been installed to. After
obtaining a copy, set an environment variable called `BNG_HOME` that contains
the path to your local installation's main directory -- the same that contains
the `EULA.pdf` file or you can add your local installation path to the `SIMULATOR_HOME_DIR` parameter in the `wcarace_ex.py` example.

## Usage

BeamNG.gym registers the environments with the OpenAI Gym registry, so after
the initial setup, the environments can be created using the factory method and
the respective environment's ID. For example:

```python
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
```

## Environments

Currently, the only environment is a time attack on the race track in the
West Coast USA level of BeamNG.drive. New environments are being developed.

### WCA Race

In this setting, the car spawns at the starting line of the race track in
West Coast USA and has to race one lap. A detailled description of the
observation and actions can be found in the documentation of the respective
class [WCARaceGeometry](https://github.com/BeamNG/BeamNG.gym).
