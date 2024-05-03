from typing import Any
import math
import numpy as np
from numpy import ndarray
from gymnasium import spaces

from .gym_continuous_maze import ContinuousMaze, get_intersect

class ContinuousLidarMaze(ContinuousMaze):
    """ An adaptation of Continuous Maze to 

        1. Add Lidar Sensor Observations
        2. Death on collision with a wall
    """
    lidar_range = 1.0

    action_space = spaces.Box(-1, 1, (2,))
    observation_space = spaces.Box(
        low = np.array([-12,-12, 0, 0, 0, 0, 0, 0, 0, 0]),
        high = np.array([12, 12, 1, 1, 1, 1, 1, 1, 1, 1])
    )

    rays = np.array([
        [1.0, 0],
        [1/math.sqrt(2), 1/math.sqrt(2)],
        [0.0, 1.0],
        [-1/math.sqrt(2), 1/math.sqrt(2)],
        [-1.0, 0.0],
        [-1/math.sqrt(2), -1/math.sqrt(2)],
        [0.0, -1.0],
        [1/math.sqrt(2), -1/math.sqrt(2)]
    ])

    def get_lidar_data(self, pos: ndarray):
        rays = self.lidar_range * self.rays
        distances = np.ones((8,)) * self.lidar_range
        for i, ray in enumerate(rays):
            new_pos = pos + ray
            for wall in self.walls:
                intersection = get_intersect(wall[0], wall[1], pos, new_pos)
                if intersection is not None:
                    intersection = intersection - pos
                    distance = np.linalg.norm(intersection)
                    distances[i] = min(distances[i], distance)
        return distances
    
    def reset(self, *args, **kwargs) -> tuple[ndarray, dict[str, Any]]:
        pos, info = super().reset(*args, **kwargs)
        lidar = self.get_lidar_data(self.pos)
        obs = np.concatenate((pos, lidar))
        return obs, info

    def step(self, action: ndarray) -> tuple[ndarray, float, bool, bool, dict]:
        # Movement
        new_pos = self.pos + action
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], self.pos, new_pos)
            if intersection is not None:
                # Edit 1: Death
                self.pos = intersection
                lidar = self.get_lidar_data(self.pos)
                obs = np.concatenate((self.pos, lidar))
                return obs, 0.0, True, False, {}
        self.pos = new_pos
        self.all_pos.append(self.pos.copy())

        # Edit 2: Lidar
        lidar = self.get_lidar_data(self.pos)
        obs = np.concatenate((self.pos, lidar))

        return obs, 0.0, False, False, {}