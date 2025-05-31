import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from scipy.interpolate import splprep, splev
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
import time
import psutil
import os
import pandas as pd
from skimage.morphology import skeletonize
import glob
from noise import pnoise1

# Assuming PathTraceEnv is defined as in your previous context
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

    def __init__(self, width=450, height=450, road_width=120, max_steps=10000):
        super().__init__()

        self.width = width
        self.height = height
        self.road_width = road_width
        self.step_size = max(1, self.height // 60)
        self.agent_radius = max(1, self.height // 50)
        self.scroll_speed = 2
        self.survival_reward = 1.0
        self.no_movement_penalty = -1.0
        self.off_road_penalty = -2.0
        self.alignment_window = 10
        self.max_steps = max_steps
        self.road_offset_changed = True
        self.cached_road_img = None
        self.cached_skeleton = None

        self.log_data = {
            "step": [],
            "agent_pos_y": [],
            "agent_pos_x": [],
            "reward": [],
            "penalty": [],
            "alignment_bonus": [],
            "distance_bonus": []
        }

        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.agent_color = (0, 0, 255)
        self.road_color = (180, 180, 180)
        self.centerline_color = (0, 0, 255)
        self.bg_color = (0, 0, 0)

        self.actions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

        self._generate_road()
        self.reset()

    def _generate_road(self):
        # Parameters
        num_points = self.max_steps * self.scroll_speed  # Total points in the curve
        x_vals = np.linspace(0, self.width * 2, num_points, dtype=np.float64)
        y_vals = np.zeros(num_points, dtype=np.float64)
        
        # Adjust amplitude and frequency for more pronounced S-curves
        amplitude = self.height // 2  # Larger swings for more dramatic curves
        base_frequency = 0.02  # Higher frequency for tighter S-curves
        
        # Generate y-values using a combination of sine waves
        for i in range(num_points):
            y_vals[i] = self.height // 2 + amplitude * (
                0.7 * np.sin(base_frequency * x_vals[i]) +  # Primary S-curve
                0.3 * np.sin(2.5 * base_frequency * x_vals[i]) +  # Secondary wave for variation
                0.1 * np.sin(4 * base_frequency * x_vals[i])  # Tertiary wave for subtle variation
            )
        
        # Clip y-values to stay within screen bounds
        y_vals = np.clip(y_vals, 0, self.height - 1)
        
        # Store the road curve
        self.road_curve = list(zip(x_vals.astype(int), y_vals.astype(int)))
        
        self.road_offset = 0
        self.road_offset_changed = True

    def _scroll_road(self):
        self.road_offset += self.scroll_speed
        if self.road_offset + self.width >= len(self.road_curve):
            self.road_offset = len(self.road_curve) - self.width - 1
        self.road_points = [self.road_curve[i + self.road_offset][1] for i in range(self.width)]
        self.road_offset_changed = True

    def _skeletonize_centerline(self, road_img):
        gray = cv2.cvtColor(road_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        binary = binary // 255
        skeleton = skeletonize(binary).astype(np.uint8) * 255
        return skeleton

    def _get_obs(self):
        if not hasattr(self, 'cached_road_img') or self.road_offset_changed:
            self.frame[:] = self.bg_color
            for x, y in enumerate(self.road_points):
                y1 = max(0, y - self.road_width // 2)
                y2 = min(self.height, y + self.road_width // 2 + 1)
                self.frame[y1:y2, x] = self.road_color
            self.cached_road_img = self.frame.copy()
            self.cached_skeleton = self._skeletonize_centerline(self.cached_road_img)
            self.road_offset_changed = False

        self.frame = self.cached_road_img.copy()
        self.frame[self.cached_skeleton == 255] = self.centerline_color

        ay, ax = self.agent_pos
        cv2.circle(self.frame, (ax, ay), self.agent_radius, self.agent_color, thickness=-1)

        image = cv2.resize(self.frame.copy(), (128, 128), interpolation=cv2.INTER_AREA)
        return image

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(42)

        self.frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self.agent_pos = [self.height // 2, 375]
        self.prev_agent_pos = self.agent_pos.copy()
        self.done = False
        self.total_steps = 0
        self.alignment_history = []
        self.cached_road_img = None
        self.cached_skeleton = None
        self.road_offset_changed = True

        self.log_data = {
            "step": [],
            "agent_pos_y": [],
            "agent_pos_x": [],
            "reward": [],
            "penalty": [],
            "alignment_bonus": [],
            "distance_bonus": []
        }

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

        reward, penalty = 0.0, 0.0

        if distance_from_center > self.road_width // 2 or ax <= 0:
            self.done = True
            penalty = self.off_road_penalty
            reward = penalty

            self.log_data["step"].append(self.total_steps)
            self.log_data["agent_pos_y"].append(ay)
            self.log_data["agent_pos_x"].append(ax)
            self.log_data["reward"].append(reward)
            self.log_data["penalty"].append(penalty)
            self.log_data["alignment_bonus"].append(0.0)
            self.log_data["distance_bonus"].append(0.0)

            self._save_logs()
            return self._get_obs(), reward, True, False, {}

        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)
        angle_diff = np.arccos(np.clip(np.dot(agent_dir, road_dir), -1.0, 1.0))
        alignment_bonus = (1 - (angle_diff / np.pi)) * 5.0
        distance_bonus = max(1.0 - (distance_from_center / (self.road_width / 2)), 0) * 7.0

        self.alignment_history.append(alignment_bonus / 5.0)
        if len(self.alignment_history) > self.alignment_window:
            self.alignment_history.pop(0)
        cumulative_alignment_bonus = np.mean(self.alignment_history) * 0.5 if self.alignment_history else 0.0

        dx = self.agent_pos[1] - self.prev_agent_pos[1] + self.scroll_speed
        progress_bonus = max(0, dx) * 0.1

        reward = self.survival_reward + alignment_bonus + distance_bonus + cumulative_alignment_bonus + progress_bonus

        movement = np.linalg.norm(np.array(self.agent_pos) - np.array(self.prev_agent_pos))
        if movement < 1e-3:
            penalty = self.no_movement_penalty
            reward += penalty

        if np.isnan(reward):
            raise ValueError(f"NaN detected in reward: survival={self.survival_reward}, alignment={alignment_bonus}, "
                             f"distance={distance_bonus}, cumulative={cumulative_alignment_bonus}, "
                             f"progress={progress_bonus}, penalty={penalty}")
        
        distance_penalty = -(distance_from_center / (self.road_width / 2)) * 0.5  # Small penalty for being far from center
        reward += distance_penalty
        
        self.log_data["step"].append(self.total_steps)
        self.log_data["agent_pos_y"].append(ay)
        self.log_data["agent_pos_x"].append(ax)
        self.log_data["reward"].append(reward)
        self.log_data["penalty"].append(penalty)
        self.log_data["alignment_bonus"].append(alignment_bonus)
        self.log_data["distance_bonus"].append(distance_bonus + distance_penalty)  # Update logging

        if self.total_steps % 1000 == 0:
            self._save_logs()

        self._scroll_road()
        self.agent_pos[1] -= 1
        if self.agent_pos[1] <= -self.width // 4:
            self.done = True
            penalty = self.off_road_penalty
            reward += penalty

            self.log_data["penalty"][-1] = penalty
            self.log_data["reward"][-1] = reward

            self._save_logs()
            return self._get_obs(), reward, True, False, {}
        
        obs, reward, done, truncated, info = self._step(action)  # Adjust based on your implementation
        if not np.isfinite(reward):
            print(f"Invalid reward detected: {reward}")

        return self._get_obs(), reward, False, False, {}

    def _save_logs(self):
        df = pd.DataFrame(self.log_data)
        log_file = os.path.join(self.log_dir, f"env_log_{self.total_steps}.csv")
        df.to_csv(log_file, index=False)
        #print(f"Saved logs to {log_file}")

    def render(self):
        if not hasattr(self, 'cached_road_img') or self.road_offset_changed:
            self.frame[:] = self.bg_color
            for x, y in enumerate(self.road_points):
                y1 = max(0, y - self.road_width // 2)
                y2 = min(self.height, y + self.road_width // 2 + 1)
                self.frame[y1:y2, x] = self.road_color
            self.cached_road_img = self.frame.copy()
            self.cached_skeleton = self._skeletonize_centerline(self.cached_road_img)
            self.road_offset_changed = False

        self.frame = self.cached_road_img.copy()
        self.frame[self.cached_skeleton == 255] = self.centerline_color

        ay, ax = self.agent_pos
        cv2.circle(self.frame, (ax, ay), self.agent_radius, self.agent_color, thickness=-1)

        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)

        arrow_length = 10
        end_agent = (int(ax + agent_dir[0] * arrow_length), int(ay + agent_dir[1] * arrow_length))
        end_road = (int(ax + road_dir[0] * arrow_length), int(ay + agent_dir[1] * arrow_length))

        cv2.arrowedLine(self.frame, (ax, ay), end_agent, (255, 0, 0), 1, tipLength=0.3)
        cv2.arrowedLine(self.frame, (ax, ay), end_road, (0, 255, 0), 1, tipLength=0.3)

        display_img = cv2.resize(self.frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("PathTrace", display_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._save_logs()

if __name__ == '__main__':
    # Initialize environment
    env = PathTraceEnv()
    check_env(env, warn=True)

    # Wrap for image input with 2 parallel environments
    vec_env = SubprocVecEnv([lambda: PathTraceEnv() for _ in range(2)])
    vec_env = VecTransposeImage(vec_env)

    # Training parameters
    MODEL_PATH = "D:/RL_Project/run/Improved_road/best_model"
    TOTAL_TIMESTEPS = 500000

    # Check if a saved model exists
    model_file_path = f"{MODEL_PATH}.zip"
    if os.path.exists(model_file_path):
        print(f"Loading existing model from {model_file_path}...")
        model = PPO.load(MODEL_PATH, env=vec_env, verbose=1, tensorboard_log="./Improved_road/")
        timesteps_trained = model.num_timesteps
        remaining_timesteps = max(0, TOTAL_TIMESTEPS - timesteps_trained)
        print(f"Continuing training for {remaining_timesteps} timesteps...")
    else:
        print(f"No existing model found. Starting training from scratch...")
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./Improved_road/",
            learning_rate=3e-4,
            ent_coef=0.05,
            clip_range=0.2,
            batch_size=128,
            n_epochs=30,
            vf_coef=1.0
        )
        remaining_timesteps = TOTAL_TIMESTEPS

    # Train for the remaining timesteps
    start_time = time.time()
    if remaining_timesteps > 0:
        model.learn(total_timesteps=remaining_timesteps, reset_num_timesteps=False)
        # Save the model once after training
        model.save(MODEL_PATH)
        print(f"Model saved to {model_file_path}")
    else:
        print(f"Model already trained for {timesteps_trained} timesteps. No further training needed.")

    print(f"Training time: {time.time() - start_time} seconds")