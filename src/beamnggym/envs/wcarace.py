'''
Defines the WCARaceGeometry gymnasium environment.
'''
from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.misc.quat import angle_to_quat
from beamngpy.sensors import Damage, Electrics
from shapely import affinity
from shapely.geometry import LinearRing, LineString, Point, Polygon


def normalise_angle(angle):
    """
    Normalize an angle to be within the range [0, 2π).

    Args:
        angle (float): The angle in radians to be normalized.

    Returns:
        float: The normalized angle within the range [0, 2π).
    """
    if angle < 0:
        angle += np.pi * 2
    return angle


def calculate_curvature(points, idx):
    """
    Calculate the curvature at a given point in a sequence of points.

    Args:
        points: points used to calculate the curvature.
        idx (int): The index of the point at which to calculate the curvature.
               idx cannot be the first or last index in the list.

    Returns:
        float: The curvature at the given point.

    Raises:
        ValueError: If `idx` is not between 1 and len(points) - 1.
    """
    if not 0 < idx < len(points) - 1:
        raise ValueError('idx must be between 0 and len(points) - 1')
    p1 = points[idx - 1]
    p2 = points[idx + 0]
    p3 = points[idx + 1]
    eps = 1e-8
    curvature = 2 * (
        (p2[0] - p1[0]) * (p3[1] - p2[1]) -
        (p2[1] - p1[1]) * (p3[0] - p2[0])) / (np.sqrt(
            (np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1])) *
            (np.square(p3[0] - p2[0]) + np.square(p3[1] - p2[1])) *
            (np.square(p1[0] - p3[0]) + np.square(p1[1] - p3[1]))) + eps
    )
    return curvature


def calculate_inclination(points, idx: int) -> float:
    """
    Calculate the inclination between two points.

    This function calculates the difference in the z-coordinate between the point at index `idx + 1` and the point at
    index `idx - 1` in the given list of points.

    Args:
        points: the points used to calculate the inclination.
        idx (int): The index of the point for which the inclination is to be calculated.

    Returns:
        float: The inclination between the points at index `idx + 1` and `idx - 1`.

    Raises:
        ValueError: If `idx` is not between 1 and len(points) - 1.
    """
    if not 0 < idx < len(points) - 1:
        raise ValueError('idx must be between 0 and len(points) - 1')
    p1 = points[idx - 1]
    p3 = points[idx + 1]
    inclination = p3[2] - p1[2]
    return inclination


class WCARaceGeometry(gym.Env):
    """
    A gymnasium environment for the race circuit at the WCUSA map in BeamNG.tech.
    """
    sps = 50
    rate = 5

    front_dist = 800  # Distance ahead of the vehicle to generate spline observations
    front_step = 100  # Number of steps ahead of the car to generate spline observations
    front_factor = front_dist / front_step
    trail_dist = 104  # Distance behind the vehicle to generate spline observations
    trail_step = 13  # Number of trailing steps behind the car to generate spline observations
    trail_factor = trail_dist / trail_step

    starting_proj = 1710
    max_damage = 100

    def __init__(self, host='localhost', port=64256, home=None):
        self.steps = WCARaceGeometry.sps // WCARaceGeometry.rate
        self.host = host
        self.port = port
        self.home = home

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.episode_steps = 0
        self.spine = None
        self.l_edge = None
        self.r_edge = None
        self.polygon = None

        self.bng = BeamNGpy(host=self.host, port=self.port, home=self.home)
        self.bng.open()

        self.vehicle = Vehicle('racecar', model='sunburst', license='BEAMNG',
                               color='red',
                               part_config='vehicles/sunburst/hillclimb.pc')

        electrics = Electrics()
        damage = Damage()
        self.vehicle.sensors.attach('electrics', electrics)
        self.vehicle.sensors.attach('damage', damage)

        scenario = Scenario('west_coast_usa', 'wca_race_geometry_v0')
        scenario.add_vehicle(self.vehicle, pos=(394.5, -247.925, 145.25), rot_quat=angle_to_quat((0, 0, 90)))

        scenario.make(self.bng)

        self.bng.open(launch=True)
        self.bng.settings.set_deterministic(WCARaceGeometry.sps)
        self.bng.scenario.load(scenario)

        self._build_racetrack()

        self.observation = None
        self.last_observation = None
        self.last_spine_proj = None

        self.bng.scenario.start()
        self.bng.control.pause()

    def __del__(self):
        """
        Destructor method that ensures the BeamNG simulator instance is properly closed when the object is deleted.
        """
        self.bng.close()

    def _build_racetrack(self):
        """
        Builds the racetrack by extracting the Decal Road (2D spline) and road edges from the scenario and creating
        geometric shapes to represent the left edge, right edge, and spine of the track.
        """
        roads = self.bng.scenario.get_roads()
        # This is the persistent ID of the race circuit at the WCUSA map
        RACETRACK_PID = '064a5d03-61d1-4ed7-9136-905b40928f01'
        track_id, _ = next(filter(lambda road: road[1]['persistentId'] == RACETRACK_PID, roads.items()))
        track = self.bng.scenario.get_road_edges(track_id)
        l_vtx = []
        s_vtx = []
        r_vtx = []
        for edges in track:
            r_vtx.append(edges['right'])
            s_vtx.append(edges['middle'])
            l_vtx.append(edges['left'])

        self.spine = LinearRing(s_vtx)
        self.r_edge = LinearRing(r_vtx)
        self.l_edge = LinearRing(l_vtx)

        r_vtx = [v[0:2] for v in r_vtx]
        l_vtx = [v[0:2] for v in l_vtx]
        self.polygon = Polygon(l_vtx, holes=[r_vtx])

    def _action_space(self):
        """
        Defines the action space for the environment.

        The action space is a continuous space represented by a Box with two dimensions. Each dimension can take values
        in the range [-1, 1].

        Returns:
            gym.spaces.Box: A Box object representing the action space.
        """
        action_lo = [-1., -1.]
        action_hi = [+1., +1.]
        return spaces.Box(np.array(action_lo), np.array(action_hi),
                          dtype=float)

    def _observation_space(self):
        # n vertices of left and right polylines ahead and behind, 3 floats per
        # vtx
        scope = WCARaceGeometry.trail_step + WCARaceGeometry.front_step
        obs_lo = [-np.inf, ] * scope * 3
        obs_hi = [np.inf, ] * scope * 3
        obs_lo.extend([
            -np.inf,     # Distance to left edge
            -np.inf,     # Distance to right edge
            -2 * np.pi,  # Inclination
            -2 * np.pi,  # Angle
            -2 * np.pi,  # Vertical angle
            -np.inf,     # Spine speed
            0,           # RPM
            -1,          # Gear
            0,           # Throttle
            0,           # Brake
            -1.0,        # Steering
            0,           # Wheel speed
            -np.inf,     # Altitude
        ])
        obs_hi.extend([
            np.inf,      # Distance to left edge
            np.inf,      # Distance to right edge
            2 * np.pi,   # Inclincation
            2 * np.pi,   # Angle
            2 * np.pi,   # Vertical angle
            np.inf,      # Spine speed
            np.inf,      # RPM
            8,           # Gear
            1.0,         # Throttle
            1.0,         # Brake
            1.0,         # Steering
            np.inf,      # Wheel speed
            np.inf,      # Altitude
        ])
        return spaces.Box(np.array(obs_lo), np.array(obs_hi),
                          dtype=float)

    def _make_commands(self, action):
        """
        Maps and actuates the gymnasium action to a BeamNGpy control command.

        Args:
            action (list or tuple): A sequence containing steering and
                                    throttle values.
                - action[0]: Steering value (float) where negative values indicate left turn and positive values
                             indicate right turn.
                - action[1]: Throttle value (float) where positive values indicate acceleration and negative values
                             indicate braking.
        """
        brake = 0
        throttle = action[1]
        steering = action[0]
        if throttle < 0:
            brake = -throttle
            throttle = 0

        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)

    def _project_vehicle(self, pos):
        """
        Projects a vehicle's position onto the left edge, right edge, and spine of the track.

        Args:
            pos (tuple): The position of the vehicle to be projected.

        Returns:
            tuple: A tuple containing the projected positions on the left edge, spine, and right edge.
        """
        r_proj = self.r_edge.project(pos)
        r_proj = self.r_edge.interpolate(r_proj)
        l_proj = self.l_edge.project(r_proj)
        l_proj = self.l_edge.interpolate(l_proj)
        s_proj = self.spine.project(r_proj)
        s_proj = self.spine.interpolate(s_proj)
        return l_proj, s_proj, r_proj

    def _get_vehicle_angles(self, vehicle_pos, spine_seg):
        spine_beg = spine_seg.coords[+0]
        spine_end = spine_seg.coords[-1]
        spine_angle = np.arctan2(spine_end[1] - spine_beg[1],
                                 spine_end[0] - spine_beg[0])
        vehicle_angle = self.vehicle.state['dir'][0:2]
        vehicle_angle = np.arctan2(vehicle_angle[1], vehicle_angle[0])

        vehicle_angle = normalise_angle(vehicle_angle - spine_angle)
        vehicle_angle -= np.pi

        elevation = np.arctan2(spine_beg[2] - spine_end[2], spine_seg.length)
        vehicle_elev = self.vehicle.state['dir']
        vehicle_elev = np.arctan2(vehicle_elev[2],
                                  np.linalg.norm(vehicle_elev))

        return vehicle_angle, vehicle_elev, elevation

    def _wrap_length(self, target):
        length = self.spine.length
        while target < 0:
            target += length
        while target > length:
            target -= length
        return target

    def _gen_track_scope_loop(self, it, fn, base, s_scope, s_width):
        for step in it:
            distance = base + fn(step)
            distance = self._wrap_length(distance)
            s_proj = self.spine.interpolate(distance)
            s_scope.append(s_proj)
            l_proj = self.l_edge.project(s_proj)
            l_proj = self.l_edge.interpolate(l_proj)
            r_proj = self.r_edge.project(s_proj)
            r_proj = self.r_edge.interpolate(r_proj)
            width = l_proj.distance(r_proj)
            s_width.append(width)

    def _gen_track_scope(self, vehicle_pos, spine_seg):
        s_scope = []
        s_width = []

        self.front_step_fn = lambda step: +WCARaceGeometry.front_factor * step
        self.trail_step_fn = lambda step: -WCARaceGeometry.trail_factor * step

        # Project the vehicle's position onto the spine of the track
        base = self.spine.project(vehicle_pos)

        # Add the track spline points behind the vehicle to the scope
        it = range(WCARaceGeometry.trail_step, 0, -1)
        self._gen_track_scope_loop(it, self.trail_step_fn, base, s_scope, s_width)

        # Add the track point directly at the vehicle to the scope
        it = range(1)  # TODO: Is this redundant due to the ahead points below?
        self._gen_track_scope_loop(it, lambda x: x, base, s_scope, s_width)

        # Add the track spline points ahead of the vehicle to the scope
        it = range(WCARaceGeometry.front_step + 1)
        self._gen_track_scope_loop(it, self.trail_step_fn, base, s_scope, s_width)

        s_proj = self.spine.interpolate(base)
        offset = (-s_proj.x, -s_proj.y, -s_proj.z)
        s_line = LineString(s_scope)
        s_line = affinity.translate(s_line, *offset)

        spine_beg = spine_seg.coords[+0]
        spine_end = spine_seg.coords[-1]
        direction = [spine_end[i] - spine_beg[i] for i in range(3)]
        # Get the angle of the spine in the XY plane
        angle = np.arctan2(direction[1], direction[0]) + np.pi / 2

        s_line = affinity.rotate(s_line, -angle, origin=(0, 0),
                                 use_radians=True)

        ret = list()
        s_scope = s_line.coords
        for idx in range(1, len(s_scope) - 1):
            curvature = calculate_curvature(s_scope, idx)
            inclination = calculate_inclination(s_scope, idx)
            width = s_width[idx]
            ret.append(curvature)
            ret.append(inclination)
            ret.append(width)

        return ret

    def _spine_project_vehicle(self, vehicle_pos):
        """
        Projects a vehicle's position onto the spine of the track and returns the distance remaining to the end of the
        track.

        Args:
            vehicle_pos (float): The position of the vehicle to be projected.

        Returns:
            float: The adjusted projection of the vehicle's position on the spine.
        """
        proj = self.spine.project(vehicle_pos) - WCARaceGeometry.starting_proj
        if proj < 0:
            proj += self.spine.length
        # Distance remaining to the end of the track
        proj = self.spine.length - proj
        return proj

    def _get_spine_speed(self, vehicle_pos, vehicle_dir, spine_seg):
        spine_beg = spine_seg.coords[0]
        future_pos = Point(vehicle_pos.x + vehicle_dir[0],
                           vehicle_pos.y + vehicle_dir[1],
                           vehicle_pos.z + vehicle_dir[2])
        spine_end = self.spine.project(future_pos)
        spine_end = self.spine.interpolate(spine_end)
        return spine_end.distance(Point(*spine_beg))

    def _make_observation(self, sensors):
        electrics = sensors['electrics']

        vehicle_dir = self.vehicle.state['dir']
        vehicle_pos = self.vehicle.state['pos']
        vehicle_pos = Point(*vehicle_pos)

        # Get the spine segment the vehicle is on
        spine_beg = self.spine.project(vehicle_pos)
        spine_end = spine_beg
        spine_end += WCARaceGeometry.front_factor
        spine_beg = self.spine.interpolate(spine_beg)
        spine_end = self.spine.interpolate(spine_end)
        spine_seg = LineString([spine_beg, spine_end])

        spine_speed = self._get_spine_speed(vehicle_pos, vehicle_dir,
                                            spine_seg)

        l_dist = self.l_edge.distance(vehicle_pos)
        r_dist = self.r_edge.distance(vehicle_pos)

        angle, vangle, elevation = self._get_vehicle_angles(vehicle_pos,
                                                            spine_seg)

        l_proj, s_proj, r_proj = self._project_vehicle(vehicle_pos)
        s_scope = self._gen_track_scope(vehicle_pos, spine_seg)

        obs = list()
        obs.extend(s_scope)
        obs.append(l_dist)
        obs.append(r_dist)
        obs.append(elevation)
        obs.append(angle)
        obs.append(vangle)
        obs.append(spine_speed)
        obs.append(electrics['rpm'])
        obs.append(electrics['gear_index'])
        obs.append(electrics['throttle'])
        obs.append(electrics['brake'])
        obs.append(electrics['steering'])
        obs.append(electrics['wheelspeed'])
        obs.append(electrics['altitude'])

        return np.array(obs)

    def _compute_reward(self, sensors):
        """
        Computes the reward based on the vehicle's damage and progression along the spine of the track.

        Args:
            sensors (dict): A dictionary containing sensor data, including 'damage'.

        Returns:
            tuple: A tuple containing:
                - score (float): The computed reward score.
                - truncated (bool): Whether the episode is truncated.
                - terminated (bool): Whether the episode is terminated.
        """
        damage = sensors['damage']
        vehicle_pos = self.vehicle.state['pos']
        vehicle_pos = Point(*vehicle_pos)

        # If the damage exceeds the maximum allowed damage, the episode is truncated.
        if damage['damage'] > WCARaceGeometry.max_damage:
            return -1, True, False

        # If the vehicle is outside the track polygon, the episode is truncated.
        if not self.polygon.contains(Point(vehicle_pos.x, vehicle_pos.y)):
            return -1, True, False

        score, truncated, terminated = -1, False, False
        spine_proj = self._spine_project_vehicle(vehicle_pos)  # Distance remaining to the end of the track
        if self.last_spine_proj is not None:  # If this is not the first step
            diff = spine_proj - self.last_spine_proj
            if diff <= 0:  # If the vehicle is moving forwards
                if diff > -0.2:  # If the vehicle is moving slowly forwards
                    return -1, False, False
                else:  # If the vehicle is moving quickly forwards
                    score, truncated, terminated = diff / self.steps, False, False
            elif np.abs(diff) > self.spine.length * 0.95:  # If the vehicle has completed at least 95% of the track
                score, truncated, terminated = 1, False, True
            else:  # If the vehicle is moving backwards and has not completed at least 95% of the track
                score, truncated, terminated = -1, True, False
        self.last_spine_proj = spine_proj
        return score, truncated, terminated

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Resets the environment to an initial state and returns an initial observation.

        Args:
            seed (int | None): The seed for random number generation. Default is None.
            options (dict[str, Any] | None): Additional options for resetting the environment. Default is None.

        Returns:
            tuple:
                - observation (any): The initial observation.
                - info (dict): Additional information, currently an empty dictionary.
        """
        super().reset(seed=seed, options=options)

        self.episode_steps = 0
        self.vehicle.control(throttle=0.0, brake=0.0, steering=0.0)
        self.bng.scenario.restart()
        self.bng.control.step(30)
        self.bng.control.pause()
        self.vehicle.set_shift_mode('realistic_automatic')
        self.vehicle.control(gear=2)
        self.vehicle.sensors.poll()
        sensors = self.vehicle.sensors
        self.observation = self._make_observation(sensors)
        vehicle_pos = self.vehicle.state['pos']
        vehicle_pos = Point(*vehicle_pos)
        self.last_spine_proj = self._spine_project_vehicle(vehicle_pos)
        return self.observation, {}

    def advance(self):
        """
        Advances the BeamNG simulation by the number of steps specified in the `steps` attribute.
        """
        self.bng.step(self.steps, wait=True)

    def observe(self):
        """
        Polls the vehicle's sensors to update their state and then generates a new observation.

        Returns:
            tuple:
                - new_observation: The newly created observation based on the sensor data.
                - sensors: The sensor objects providing the latest sensor data.
        """
        self.vehicle.sensors.poll()
        sensors = self.vehicle.sensors
        new_observation = self._make_observation(sensors)
        return new_observation, sensors

    def step(self, action: list) -> tuple:
        """
        Execute one step in the environment with the given action and returns the new observation, reward, and episode
        end statuses.

        Args:
            action (list): A list of action values to be taken by the agent. The values are clipped between -1 and 1.

        Returns:
        tuple: A tuple containing:
            - observation (any): The new observation after taking the action.
            - score (float): The reward obtained after taking the action.
            - terminated (bool): Whether the episode has terminated.
            - truncated (bool): Whether the episode has been truncated.
            - info (dict): Additional information, currently an empty dictionary.
        """
        action = [*np.clip(action, -1, 1), action[0], action[1]]
        action = [float(v) for v in action]

        self.episode_steps += 1

        self._make_commands(action)
        self.advance()
        new_observation, sensors = self.observe()
        if self.observation is not None:
            self.last_observation = self.observation
        self.observation = new_observation
        score, truncated, terminated = self._compute_reward(sensors)

        print(f' A: {action[2]:5.2f}  B: {action[3]:5.2f} '
              f' S: {action[0]:5.2f}  T: {action[1]:5.2f}  R: {score:5.2f}')

        if truncated:
            print('Episode truncated')

        if terminated:
            print('Episode terminated')

        return self.observation, score, terminated, truncated, {}
