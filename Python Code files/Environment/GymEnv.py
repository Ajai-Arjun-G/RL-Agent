import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from scipy.interpolate import splprep, splev
import cv2
from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


class PathTraceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    class Actions(Enum):
        Left = 0
        Right = 1
        Up = 2
        Down = 3
        UpLeft = 4
        UpRight = 5
        DownLeft = 6
        DownRight = 7

    def __init__(self, width=600, height=600, road_width=150):
        super().__init__()

        self.width = width
        self.height = height
        self.total_steps = 0
        self.road_width = road_width
        self.step_size = max(1, self.height // 50)
        self.agent_radius = max(1, self.height // 50)

        self.agent_color = (255, 0, 0)    # Red
        self.road_color = (180, 180, 180) # Light Gray
        self.bg_color = (0, 0, 0)         # Black

        self.actions = [
            (-1, 0),   # up
            (-1, 1),   # up-right
            (0, 1),    # right
            (1, 1),    # down-right
            (1, 0),    # down
            (1, -1),   # down-left
            (0, -1),   # left
            (-1, -1),  # up-left
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(96, 96, 3),
            dtype=np.uint8
        )


        self._generate_road_curve()
        self.window = None
        self.clock = None
        self.reset()

    def _generate_road_curve(self):
        control_x = np.linspace(0, self.width * 2, 8)
        control_y = self.height // 2 + np.random.randint(-self.height // 4, self.height // 4, size=8)
        tck, _ = splprep([control_x, control_y], s=0)
        u = np.linspace(0, 1, num=self.width * 2)
        x_vals, y_vals = splev(u, tck)
        self.road_curve = list(zip(x_vals.astype(int), y_vals.astype(int)))
        self.road_offset = 0

    def _scroll_road(self):
        self.road_offset += 1
        if self.road_offset + self.width >= len(self.road_curve):
            self._generate_road_curve()
            self.road_offset = 0

        self.road_points = [self.road_curve[i + self.road_offset][1] for i in range(self.width)]

    def _get_obs(self):
        self.frame[:] = self.bg_color
        for x, y in enumerate(self.road_points):
            y1 = max(0, y - self.road_width // 2)
            y2 = min(self.height, y + self.road_width // 2 + 1)
            self.frame[y1:y2, x] = self.road_color
        ay, ax = self.agent_pos
        if 0 <= ay < self.height and 0 <= ax < self.width:
            cv2.circle(self.frame, (ax, ay), self.agent_radius, self.agent_color, thickness=-1)
        
        # Resize the full frame to match CNN input size
        resized_obs = cv2.resize(self.frame.copy(), (96, 96), interpolation=cv2.INTER_AREA)
        return resized_obs


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self.agent_pos = [self.height // 2, 500]  # [y, x]
        self.steps_stationary = 0
        #print(self.total_steps)
        self.total_steps = 0
        self.done = False
        self._generate_road_curve()
        self._scroll_road()
        return self._get_obs(), {}


    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
        
        self.total_steps += 1
        dy, dx = self.actions[action]
        dy *= self.step_size
        dx *= self.step_size

        self.agent_pos[0] = np.clip(self.agent_pos[0] + dy, 0, self.height - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dx, 0, self.width - 1)

        ax = self.agent_pos[1]
        ay = self.agent_pos[0]

        road_center = self.road_points[ax]
        distance_from_center = abs(road_center - ay)

        # Penalty if off-road
        if distance_from_center > self.road_width // 2 or ax <= 0:
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        # Reward shaping based on distance from center
        reward = max(1.0 - (distance_from_center / (self.road_width / 2)), 0)

        # Scroll road and agent
        self._scroll_road()
        self.agent_pos[1] -= 1  # scroll left with road
        if self.agent_pos[1] <= 0:
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        return self._get_obs(), reward, False, False, {}

    def render(self):
        img = self._get_obs()
        display_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("PathTrace", display_img)
        cv2.waitKey(1)


    def close(self):
        cv2.destroyAllWindows()

register(
    id="PathTrace-v0",
    entry_point="your_module_path:PathTraceEnv",
)

env = PathTraceEnv()

# Optional: check the environment
check_env(env, warn=True)

# Wrap with DummyVecEnv for image input
vec_env = DummyVecEnv([lambda: env])

#model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./ppo_pathtrace_tensorboard/")

#model.learn(total_timesteps=500000)

#model.save("ppo_pathtrace_cnn")

from stable_baselines3 import PPO

env = PathTraceEnv()
model = PPO.load("ppo_pathtrace_cnn")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
print(env.unwrapped.total_steps)
env.close()


