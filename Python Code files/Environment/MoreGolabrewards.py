import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from scipy.interpolate import splprep, splev
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from skimage.morphology import skeletonize
import os
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import time

class PathTraceEnv(gym.Env):
    """Custom Gymnasium environment where an agent navigates a scrolling road."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

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
        """
        Initialize the PathTraceEnv.

        Args:
            width (int): Width of the environment frame.
            height (int): Height of the environment frame.
            road_width (int): Width of the road.
        """
        super().__init__()

        # Environment parameters
        self.width = width
        self.height = height
        self.road_width = max(10, road_width)
        self.step_size = max(1, self.height // 50)
        self.agent_radius = max(1, self.height // 50)
        self.scroll_speed = 3
        self.arrow_length = 10
        self.survival_reward = 0.05
        self.no_movement_penalty = -1.0
        self.off_road_penalty = -10.0
        self.alignment_window = 10  # Window for cumulative alignment reward
        self.alignment_history = []  # Store alignment bonuses

        # Colors
        self.agent_color = (0, 0, 255)  # Red
        self.road_color = (180, 180, 180)  # Gray
        self.centerline_color = (0, 0, 255)  # Blue
        self.bg_color = (0, 0, 0)  # Black

        # Action space: 8 directional movements
        self.actions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)

        # Logging
        self.trajectory_log = []
        self.reward_log = []
        self.penalty_log = []

        # Initialize road and frame
        self.frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self._generate_road()
        self.road_offset_changed = True

    def _generate_road(self):
        """
        Generate a smooth road curve using B-spline interpolation, ensuring continuity if road exists.

        Returns:
            None
        """
        if hasattr(self, 'road_curve'):
            last_x, last_y = self.road_curve[-1]
            control_x = np.linspace(last_x, last_x + self.width * 2, 8)
            control_y = last_y + np.random.randint(-self.height // 4, self.height // 4, size=8)
        else:
            control_x = np.linspace(0, self.width * 2, 8)
            control_y = self.height // 2 + np.random.randint(-self.height // 4, self.height // 4, size=8)
        tck, _ = splprep([control_x, control_y], s=0)
        u = np.linspace(0, 1, num=self.width * self.scroll_speed)
        x_vals, y_vals = splev(u, tck)
        self.road_curve = list(zip(x_vals.astype(int), y_vals.astype(int)))
        self.road_offset = 0
        self.road_offset_changed = True

    def _scroll_road(self):
        """
        Scroll the road and regenerate if necessary.

        Returns:
            None
        """
        self.road_offset += self.scroll_speed
        if self.road_offset + self.width >= len(self.road_curve):
            self._generate_road()
        self.road_points = [self.road_curve[i + self.road_offset][1] for i in range(self.width)]
        self.road_offset_changed = True

    def _skeletonize_centerline(self, road_img):
        """
        Generate a skeletonized centerline from the road image.

        Args:
            road_img (np.ndarray): RGB image of the road.

        Returns:
            np.ndarray: Binary image of the centerline.
        """
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

        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)
        agent_dir_idx = self._vector_to_direction(agent_dir)
        road_dir_idx = self._vector_to_direction(road_dir)
        #print(f"_get_obs agent_dir_idx: {agent_dir_idx}, road_dir_idx: {road_dir_idx}")
        image = cv2.resize(self.frame.copy(), (96, 96), interpolation=cv2.INTER_AREA)
        if image.shape != (96, 96, 3):
            print(f"Unexpected image shape: {image.shape}")
            image = image.transpose(1, 2, 0) if image.shape == (96, 3, 96) else image
            assert image.shape == (96, 96, 3), f"Image shape {image.shape} not corrected"
        return image
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.

        Returns:
            tuple: Observation and info dictionary.
        """
        super().reset(seed=seed)
        self.frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        self.agent_pos = [self.height // 2, 500]
        self.prev_agent_pos = self.agent_pos.copy()
        self.done = False
        self.total_steps = 0
        self.trajectory_log = []
        self.reward_log = []
        self.penalty_log = []
        self.alignment_history = []
        self._generate_road()
        self._scroll_road()
        return self._get_obs(), {}

    def _get_direction_vector(self, from_pos, to_pos):
        """
        Calculate normalized direction vector between two points.

        Args:
            from_pos (list): Starting position [y, x].
            to_pos (list): Ending position [y, x].

        Returns:
            tuple: Normalized direction vector (dx, dy).
        """
        dy = to_pos[0] - from_pos[0]
        dx = to_pos[1] - from_pos[1]
        norm = np.linalg.norm([dx, dy])
        if norm < 1e-6:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def _get_road_tangent(self, x):
        """
        Calculate the tangent vector of the road at position x.

        Args:
            x (int): X-coordinate.

        Returns:
            tuple: Normalized tangent vector (dx, dy).
        """
        x0 = max(0, x - 2)
        x1 = min(self.width - 1, x + 2)
        y0 = self.road_points[x0]
        y1 = self.road_points[x1]
        return self._get_direction_vector((y0, x0), (y1, x1))

    def _vector_to_direction(self, vector):
        """
        Convert a direction vector to a discrete direction index.

        Args:
            vector (tuple): Normalized direction vector (dx, dy).

        Returns:
            int: Direction index (0: Up, 1: UpRight, ..., 7: UpLeft).
        """
        dx, dy = vector
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0  # Default to Up if no movement
        angle = np.arctan2(dy, dx)
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        angle_deg = np.degrees(angle)
        direction_idx = int((angle_deg + 22.5) % 360 // 45)
        return direction_idx

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): Action index from Actions enum.

        Returns:
            tuple: Observation, reward, done, truncated, info.
        """
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
            self.penalty_log.append(penalty)
            reward = penalty
            return self._get_obs(), reward, True, False, {}

        # Calculate alignment and distance bonuses
        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)
        angle_diff = np.arccos(np.clip(np.dot(agent_dir, road_dir), -1.0, 1.0))
        alignment_bonus = 1 - (angle_diff / np.pi)  # [0, 1]
        distance_bonus = max(1.0 - (distance_from_center / (self.road_width / 2)), 0)  # [0, 1]

        # Cumulative alignment reward
        self.alignment_history.append(alignment_bonus)
        if len(self.alignment_history) > self.alignment_window:
            self.alignment_history.pop(0)
        cumulative_alignment_bonus = np.mean(self.alignment_history) * 0.5 if self.alignment_history else 0.0

        # Progress bonus
        dx = self.agent_pos[1] - self.prev_agent_pos[1] + self.scroll_speed  # Adjust for scrolling
        progress_bonus = max(0, dx) * 0.01

        # Reward: survival + alignment * distance + cumulative alignment + progress
        reward = self.survival_reward + (alignment_bonus * distance_bonus * 10.0) + cumulative_alignment_bonus + progress_bonus

        # Penalty for no movement
        movement = np.linalg.norm(np.array(self.agent_pos) - np.array(self.prev_agent_pos))
        if movement < 1e-3:
            penalty = self.no_movement_penalty
            reward += penalty
            self.penalty_log.append(penalty)

        reward = np.clip(reward, -10.0, 10.0)
        self.reward_log.append(reward)
        self.trajectory_log.append((self.agent_pos[0], self.agent_pos[1]))

        # Save logs every 1000 steps
        if self.total_steps % 1000 == 0:
            np.save(f"agent_positions_{self.total_steps}.npy", np.array(self.trajectory_log))
            np.save(f"rewards_{self.total_steps}.npy", np.array(self.reward_log))
            np.save(f"penalties_{self.total_steps}.npy", np.array(self.penalty_log))

        self._scroll_road()
        self.agent_pos[1] -= 1
        if self.agent_pos[1] <= -self.width // 4:
            self.done = True
            penalty = self.off_road_penalty
            reward += penalty
            self.penalty_log.append(penalty)
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), reward, False, False, {}

    def render(self, mode="human", step=None):
        """
        Render the environment according to the specified mode.

        Args:
            mode (str): Rendering mode ("human" or "rgb_array").
            step (int, optional): Current step for labeling in human mode.

        Returns:
            np.ndarray or None: RGB array for "rgb_array" mode, None or key for "human" mode.
        """
        img = self._get_obs()["image"].copy()
        ay, ax = self.agent_pos

        agent_dir = self._get_direction_vector(self.prev_agent_pos, self.agent_pos)
        road_dir = self._get_road_tangent(ax)

        end_agent = (int(ax + agent_dir[0] * self.arrow_length), int(ay + agent_dir[1] * self.arrow_length))
        end_road = (int(ax + road_dir[0] * self.arrow_length), int(ay + road_dir[1] * self.arrow_length))

        origin = (ax, ay)
        end_agent_cv = (end_agent[0], end_agent[1])
        end_road_cv = (end_road[0], end_road[1])

        cv2.arrowedLine(img, origin, end_agent_cv, (255, 0, 0), 1, tipLength=0.3)
        cv2.arrowedLine(img, origin, end_road_cv, (0, 255, 0), 1, tipLength=0.3)

        cv2.putText(img, "Agent", (end_agent_cv[0]+3, end_agent_cv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.putText(img, "Road", (end_road_cv[0]+3, end_road_cv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        display_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

        if mode == "rgb_array":
            return display_img
        elif mode == "human":
            use_matplotlib = True
            key = -1
            try:
                cv2.imshow("PathTrace", display_img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    return key
                use_matplotlib = False
            except cv2.error:
                print("Warning: OpenCV display failed. Falling back to matplotlib.")
                use_matplotlib = True

            if use_matplotlib:
                clear_output(wait=True)
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
                plt.title(f"Step: {step}" if step is not None else "Environment View")
                plt.axis('off')
                plt.pause(1 / self.metadata["render_fps"])
                plt.show()
            return key if key != -1 else None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """
        Close the environment and display trajectory and reward heatmap.

        Returns:
            None
        """
        cv2.destroyAllWindows()
        if self.trajectory_log and self.reward_log:
            xs, ys = zip(*self.trajectory_log)
            rewards = np.array(self.reward_log)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(np.histogram2d(xs, ys, bins=32)[0], ax=ax[0], cmap="magma")
            ax[0].set_title("Agent Trajectory Heatmap")
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=32, weights=rewards)
            counts, _, _ = np.histogram2d(xs, ys, bins=32)
            heatmap = np.divide(heatmap, counts, where=counts != 0)
            sns.heatmap(heatmap, ax=ax[1], cmap="viridis")
            ax[1].set_title("Reward Gradient Heatmap")
            plt.show()
            np.save(f"agent_positions_final.npy", np.array(self.trajectory_log))
            np.save(f"rewards_final.npy", np.array(self.reward_log))
            np.save(f"penalties_final.npy", np.array(self.penalty_log))
        else:
            print("No data to plot.")


    
# Training and Inference
if __name__ == "__main__":
    from stable_baselines3.common.vec_env import VecTransposeImage

    env = PathTraceEnv()
    check_env(env, warn=True)

    vec_env = SubprocVecEnv([lambda: PathTraceEnv() for _ in range(4)])
    vec_env = VecTransposeImage(vec_env)

    TRAIN = True  # Train new model

    if TRAIN:
        eval_env = SubprocVecEnv([lambda: PathTraceEnv() for _ in range(1)])
        eval_env = VecTransposeImage(eval_env)
        eval_callback = EvalCallback(eval_env, best_model_save_path="best_model_image_cnn",
                                     log_path="./logs/", eval_freq=5000,
                                     deterministic=True, render=False)
        model_path = "ppo_pathtrace_image_cnn"
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + "_new"
            print(f"Warning: {model_path}.zip already exists. Saving as {model_path}.zip")
        model = PPO(
            policy=ActorCriticCnnPolicy,
            env=vec_env,
            verbose=1,
            tensorboard_log="./ppo_pathtrace_tensorboard/",
            device="cuda",
            n_steps=512  # Reduced to lower memory usage
        )
        obs = vec_env.reset()
        print(f"Observation image shape from vec_env: {obs.shape}")
        model.learn(total_timesteps=1000000, callback=eval_callback)
        model.save(model_path)
    else:
        model_path = "ppo_pathtrace_image_cnn"
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Model {model_path}.zip not found. Run training first.")
        model = PPO.load(model_path, device="cuda")

        env = PathTraceEnv()
        save_path = "output_frames"
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        episode = 0
        while True:
            obs, _ = env.reset()
            done = False
            step = 0
            print(f"Starting episode {episode + 1}")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                key = env.render(mode="human", step=step)
                if save_path:
                    filename = f"{save_path}/episode_{episode}_frame_{step:05d}.png"
                    cv2.imwrite(filename, env.render(mode="rgb_array"))
                if key == 27:
                    print("Exiting on user request (ESC pressed)")
                    break
                step += 1
            print(f"Episode {episode + 1} ended after {env.total_steps} steps")
            env.close()
            episode += 1
            if key == 27:
                break
        cv2.destroyAllWindows()