import gymnasium as gym

gym.register(
    id="ContinuousMaze-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze",
    max_episode_steps=100,
)

gym.register(
    id="ContinuousLidarMaze-v0",
    entry_point="gym_continuous_maze.gym_lidar_maze:ContinuousLidarMaze",
    max_episode_steps=100,
)
