import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from scipy.interpolate import splprep, splev
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from skimage.morphology import skeletonize

class PathTraceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    class Actions(Enum):
        Up = 0
        UpRight = 1
        Right = 2
        DownRight = 3
        Down = 4
        DownLeft = 5
        Left = 6
        UpLeft = 7

    def __init__(self, width=600, height=600, road_width=150):
        super().__init__()

        self.width = width
        self.height = height
        self.road_width = road_width
        self.step_size = max(1, self.height // 50)
        self.agent_radius = max(1, self.height // 50)
        self.agent_positions = []
        self.angle_differences = []
        self.scroll_speed = 3  # Adjust for faster scrolling

        self.agent_color = (0, 0, 255)
        self.road_color = (180, 180, 180)
        self.centerline_color = (0, 0, 255)
        self.bg_color = (0, 0, 0)

        self.actions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)

        self._generate_road()
        self.reset()

    def _generate_road(self):
        control_x = np.linspace(0, self.width * 2, 8)
        control_y = self.height // 2 + np.random.randint(-self.height // 4, self.height // 4, size=8)
        tck, _ = splprep([control_x, control_y], s=0)
        u = np.linspace(0, 1, num=self.width * self.scroll_speed )  # Increase multiplier
        x_vals, y_vals = splev(u, tck)
        self.road_curve = list(zip(x_vals.astype(int), y_vals.astype(int)))
        self.road_offset = 0

    def _scroll_road(self):
        self.road_offset += self.scroll_speed 
        if self.road_offset + self.width >= len(self.road_curve):
            self._generate_road()
            self.road_offset = 0
        self.road_points = [self.road_curve[i + self.road_offset][1] for i in range(self.width)]

    def _skeletonize_centerline(self, road_img):
        gray = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        binary = binary // 255
        skeleton = skeletonize(binary).astype(np.uint8) * 255
        return skeleton

    def _get_obs(self):
        self.frame[:] = self.bg_color
        for x, y in enumerate(self.road_points):
            y1 = max(0, y - self.road_width // 2)
            y2 = min(self.height, y + self.road_width // 2 + 1)
            self.frame[y1:y2, x] = self.road_color

        road_img = self.frame.copy()
        skeleton = self._skeletonize_centerline(road_img)
        self.frame[skeleton == 255] = self.centerline_color

        ay, ax = self.agent_pos
        cv2.circle(self.frame, (ax, ay), self.agent_radius, self.agent_color, thickness=-1)

        return cv2.resize(self.frame.copy(), (96, 96), interpolation=cv2.INTER_AREA)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self.agent_pos = [self.height // 2, 500]
        self.prev_agent_pos = self.agent_pos.copy()
        self.done = False
        self.total_steps = 0
        self._generate_road()
        self._scroll_road()
        return self._get_obs(), {}

    def _get_direction_vector(self, from_pos, to_pos):
        dy = to_pos[0] - from_pos[0]
        dx = to_pos[1] - from_pos[1]
        norm = np.linalg.norm([dx, dy])
        if norm == 0:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def _get_road_tangent(self, x):
        x0 = max(0, x - 2)
        x1 = min(self.width - 1, x + 2)
        y0 = self.road_points[x0]
        y1 = self.road_points[x1]
        return self._get_direction_vector((y0, x0), (y1, x1))

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        self.total_steps += 1

        dy, dx = self.actions[action]
        dy *= self.step_size
        dx *= self.step_size

        self.prev_agent_pos = self.agent_pos.copy()
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dy, 0, self.height - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dx, 0, self.width - 1)

        ax = self.agent_pos[1]
        ay = self.agent_pos[0]
        road_center = self.road_points[ax]
        distance_from_center = abs(road_center - ay)

        if distance_from_center > self.road_width // 2 or ax <= 0:
            self.done = True
            return self._get_obs(), -10.0, True, False, {}
        
        # direction is found and the angle difference is calculated here.
        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)
        angle_diff = np.arccos(np.clip(np.dot(agent_dir, road_dir), -1.0, 1.0))
        self.angle_differences.append(np.degrees(angle_diff))
        angle_loss = angle_diff / np.pi
        movement = np.linalg.norm(np.array(self.agent_pos) - np.array(self.prev_agent_pos))

        alignment_score = 1 - (angle_loss)              # [0,1], where 1 = perfectly aligned
        distance_penalty = max(1.0 - (distance_from_center / (self.road_width / 2)), 0)
        reward = (alignment_score * distance_penalty) * 10
        #reward = 1.0 - np.clip(distance_penalty, 0, 1.5)
        #reward = max(1.0 - (distance_from_center / (self.road_width / 2)), 0)

        self.agent_positions.append((self.agent_pos[0], self.agent_pos[1]))
        self.angle_differences.append(np.degrees(angle_diff))


        if self.total_steps % 1000 == 0:
            np.save("agent_positions.npy", np.array(self.agent_positions))
            np.save("angle_differences.npy", np.array(self.angle_differences))

        # reward += max(0.0, (1.0 - angle_loss)) * 0.5
        # reward *= (1.0 - angle_loss)

        if movement < 1e-3:
            reward -= 1.0  # discourage standing still
        reward = np.clip(reward, -10.0, 10.0)

        self._scroll_road()
        self.agent_pos[1] -= 1
        if self.agent_pos[1] <= 0:
            self.done = True
            return self._get_obs(), -10.0, True, False, {}

        return self._get_obs(), reward, False, False, {}

    def render(self):
        img = self._get_obs()
        ay, ax = self.agent_pos

        # Direction arrows
        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)

        arrow_length = 10
        end_agent = (int(ax + agent_dir[0] * arrow_length), int(ay + agent_dir[1] * arrow_length))
        end_road = (int(ax + road_dir[0] * arrow_length), int(ay + road_dir[1] * arrow_length))

        cv2.arrowedLine(img, (ax, ay), end_agent, (255, 0, 0), 1, tipLength=0.3)  # Blue for agent
        cv2.arrowedLine(img, (ax, ay), end_road, (0, 255, 0), 1, tipLength=0.3)   # Green for road

        display_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("PathTrace", display_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


# Initialize environment
env = PathTraceEnv()
check_env(env, warn=True)

# Wrap for image input
vec_env = DummyVecEnv([lambda: PathTraceEnv()])

# Train or load model
TRAIN = False

if TRAIN:
    model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./ppo_pathtrace_tensorboard/")
    model.learn(total_timesteps=10000)
    model.save("ppo_pathtrace_cnn")
else:
    model = PPO.load("ppo_pathtrace_cnn")

# Run the model
env = PathTraceEnv()
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs,deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
print("Total steps:", env.total_steps)
env.close()
