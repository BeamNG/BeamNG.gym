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
    if angle < 0:
        angle += np.pi * 2
    return angle


def calculate_curvature(points, idx):
    p1 = points[idx - 1]
    p2 = points[idx + 0]
    p3 = points[idx + 1]
    curvature = 2 * (
        (p2[0] - p1[0]) * (p3[1] - p2[1]) -
        (p2[1] - p1[1]) * (p3[0] - p2[0])) / (np.sqrt(
            (np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1])) *
            (np.square(p3[0] - p2[0]) + np.square(p3[1] - p2[1])) *
            (np.square(p1[0] - p3[0]) + np.square(p1[1] - p3[1]))) + 0.00000001
    )
    return curvature


def calculate_inclination(points, idx):
    p1 = points[idx - 1]
    p3 = points[idx + 1]
    inclination = p3[2] - p1[2]
    return inclination


class WCARaceGeometry(gym.Env):
    sps = 50
    rate = 5

    front_dist = 800
    front_step = 100
    trail_dist = 104
    trail_step = 13

    starting_proj = 1710
    max_damage = 100

    def __init__(self, host='localhost', port=25252):
        self.steps = WCARaceGeometry.sps // WCARaceGeometry.rate
        self.host = host
        self.port = port

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.episode_steps = 0
        self.max_episode_steps = 100000
        self.spine = None
        self.l_edge = None
        self.r_edge = None
        self.polygon = None

        front_factor = WCARaceGeometry.front_dist / WCARaceGeometry.front_step
        trail_factor = WCARaceGeometry.trail_dist / WCARaceGeometry.trail_step
        self.front = lambda step: +front_factor * step
        self.trail = lambda step: -trail_factor * step

        self.bng = BeamNGpy(self.host, self.port)
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
        self.bng.close()

    def _build_racetrack(self):
        roads = self.bng.scenario.get_roads()
        RACETRACK_PID = '064a5d03-61d1-4ed7-9136-905b40928f01'  # this is the persistent ID of the race circuit at the WCUSA map
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
        brake = 0
        throttle = action[1]
        steering = action[0]
        if throttle < 0:
            brake = -throttle
            throttle = 0

        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)

    def _project_vehicle(self, pos):
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

    def _gen_track_scope(self, pos, spine_seg):
        s_scope = []
        s_width = []

        base = self.spine.project(pos)

        it = range(WCARaceGeometry.trail_step, 0, -1)
        self._gen_track_scope_loop(it, self.trail, base, s_scope, s_width)

        it = range(1)
        self._gen_track_scope_loop(it, lambda x: x, base, s_scope, s_width)

        it = range(WCARaceGeometry.front_step + 1)
        self._gen_track_scope_loop(it, self.front, base, s_scope, s_width)

        s_proj = self.spine.interpolate(base)
        offset = (-s_proj.x, -s_proj.y, -s_proj.z)
        s_line = LineString(s_scope)
        s_line = affinity.translate(s_line, *offset)

        spine_beg = spine_seg.coords[+0]
        spine_end = spine_seg.coords[-1]
        direction = [spine_end[i] - spine_beg[i] for i in range(3)]
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
        proj = self.spine.project(vehicle_pos) - WCARaceGeometry.starting_proj
        if proj < 0:
            proj += self.spine.length
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

        spine_beg = self.spine.project(vehicle_pos)
        spine_end = spine_beg
        spine_end += WCARaceGeometry.front_dist / WCARaceGeometry.front_step
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
        damage = sensors['damage']
        vehicle_pos = self.vehicle.state['pos']
        vehicle_pos = Point(*vehicle_pos)

        # [TRUNCATE] Check if the episode reached the maximum number of steps
        if self.episode_steps >= self.max_episode_steps:
            return 0, True, False

        # [TERMINATE] Check if the vehicle is damaged
        if damage['damage'] > WCARaceGeometry.max_damage:
            return -1, False, True

        # [TERMINATE] Check if the vehicle is outside the track
        if not self.polygon.contains(Point(vehicle_pos.x, vehicle_pos.y)):
            return -1, False, True
        
        score, truncated, terminated = 0, False, False

        spine_proj = self._spine_project_vehicle(vehicle_pos)

        if self.last_spine_proj is not None:
            diff = self.last_spine_proj - spine_proj

            # [TERMINATE] Check if the vehicle is moving backwards  
            if diff < -1.0:
                return -1, False, True
            
            # [REWARD] Check if the vehicle crossed the finish line
            if np.abs(diff) > self.spine.length * 0.95:
                return 0, False, False
            
            # [REWARD] Compute the reward based on the distance covered
            score = np.clip(1 - (diff / self.spine.length), 0, 1)

            return score, False, False

        self.last_spine_proj = spine_proj
        return score, truncated, terminated

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
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
        self.bng.step(self.steps, wait=True)

    def observe(self):
        self.vehicle.sensors.poll()
        sensors = self.vehicle.sensors
        new_observation = self._make_observation(sensors)
        return new_observation, sensors

    def step(self, action):
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

        return self.observation, score, terminated, truncated, {}
