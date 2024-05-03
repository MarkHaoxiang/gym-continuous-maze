# gym-continuous-maze

Continuous maze environment integrated with OpenAI/Gym.

This fork updates the original environment to be compatible with the latest version of gymnasium, and adds ContinuousLidarMaze --- the maze adapted with local lidar distance measurements, and termination on hitting a wall.

## Installation

Clone the repository and run

```bash
cd gym-continuous-maze
pip install -e .
```

## Usage

```python
import gym
import gym_continuous_maze

env = gym.make("ContinuousMaze-v0")
env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)
    env.render()
```

![](https://raw.githubusercontent.com/qgallouedec/gym-continuous-maze/main/images/demo.png)