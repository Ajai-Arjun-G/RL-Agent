"""
The probelm to be solved: Optimize a robot to trace any given path. We define a grid in which the robot needs to trace a path.
It needs to be accurate when  it traces this path. When it follows the path till the end, the game ends. So, the robot's goal
is to trace the path as closely as possible until the end.
"""
import numpy as np
from enum import Enum
from scipy.interpolate import splprep, splev
import cv2

class Actions(Enum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
    UpLeft = 4
    UpRight = 5
    DownLeft = 6
    DownRight = 7 

class PathTrace:

    def __init__(self, width, height, road_width):
        self.width = width
        self.height = height
        self.road_width = road_width
        self.step_size = max(1, self.height // 50)

        self.agent_color = (255, 0, 0)    # Red
        self.road_color = (180, 180, 180) # Light Gray
        self.bg_color = (0, 0, 0)         # Black

        self.actions = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, 1),    # Right
            (0, -1),   # Left
            (-1, -1),  # UpLeft
            (-1, 1),   # UpRight
            (1, -1),   # DownLeft
            (1, 1),    # DownRight
        ]

        self.reset()

    def _generate_road_curve(self):
        control_x = np.linspace(0, self.width * 2, 10)
        control_y = self.height // 2 + np.random.randint(-self.height // 3, self.height // 3, size=10)
        tck, _ = splprep([control_x, control_y], s=0)
        u = np.linspace(0, 1, num=self.width * 4)
        x_vals, y_vals = splev(u, tck)
        self.road_curve = list(zip(x_vals.astype(int), y_vals.astype(int)))

    def reset(self):
        self.frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self._generate_road_curve()
        self.road_offset = 0

        self.agent_pos = [self.road_curve[10][1], 10]  # y, x start aligned on road
        self.steps_stationary = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        agent_radius = max(1, self.height // 50)
        self.frame[:] = self.bg_color

        # Draw road
        for screen_x in range(self.width):
            if self.road_offset + screen_x < len(self.road_curve):
                _, ry = self.road_curve[self.road_offset + screen_x]
                y1 = max(0, ry - self.road_width // 2)
                y2 = min(self.height, ry + self.road_width // 2 + 1)
                self.frame[y1:y2, screen_x] = self.road_color

        # Draw agent
        ay, ax = self.agent_pos
        if 0 <= ay < self.height and 0 <= ax < self.width:
            cv2.circle(self.frame, (ax, ay), agent_radius, self.agent_color, thickness=-1)

        return self.frame.copy()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True

        dy, dx = self.actions[action]
        dy *= self.step_size
        dx *= self.step_size

        old_x = self.agent_pos[1]
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dy, 0, self.height - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dx, 0, self.width - 1)

        ax = self.agent_pos[1]
        ay = self.agent_pos[0]

        # Check if agent is on the road
        curve_index = self.road_offset + ax
        if curve_index < len(self.road_curve):
            _, road_y = self.road_curve[curve_index]
            if abs(road_y - ay) > self.road_width // 2:
                self.done = True
                return self._get_obs(), -10, True
        else:
            self.done = True
            return self._get_obs(), -10, True

        # Check if agent stuck
        if ax <= 2:
            self.steps_stationary += 1
        else:
            self.steps_stationary = 0

        if self.steps_stationary > 5:
            self.done = True
            return self._get_obs(), -10, True

        # Scroll road
        self._scroll_road()
        self.agent_pos[1] -= 1  # world scrolls left

        # Agent off screen
        if self.agent_pos[1] <= 0:
            self.done = True
            return self._get_obs(), 100, True  # reward for finishing

        return self._get_obs(), 1.0, False

    def _scroll_road(self):
        self.road_offset += 1
        if self.road_offset + self.width >= len(self.road_curve):
            self._generate_road_curve()
            self.road_offset = 0

    def render(self, wait=1000):
        cv2.imshow("PathTrace", self._get_obs())
        cv2.waitKey(wait)

    def close(self):
        cv2.destroyAllWindows()

# Run it
env = PathTrace(600, 600, 50)

num_episodes = 5
for ep in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render(wait=50)
        action = np.random.randint(0, 8)
        obs, reward, done = env.step(action)
    print(f"Episode {ep + 1} done")
    cv2.waitKey(500)

env.close()
