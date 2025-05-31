import numpy as np
import os
import glob

# Directory where logs are stored (modify if saved elsewhere)
log_dir = "./"

# Find all log files
position_files = sorted(glob.glob(os.path.join(log_dir, "agent_positions_*.npy")))
reward_files = sorted(glob.glob(os.path.join(log_dir, "rewards_*.npy")))
penalty_files = sorted(glob.glob(os.path.join(log_dir, "penalties_*.npy")))

# Load data
positions = []
rewards = []
penalties = []

for pos_file, rew_file, pen_file in zip(position_files, reward_files, penalty_files):
    pos_data = np.load(pos_file)
    rew_data = np.load(rew_file)
    pen_data = np.load(pen_file)
    positions.append(pos_data)
    rewards.append(rew_data)
    penalties.append(pen_data)

# Concatenate data across all files
positions = np.concatenate(positions, axis=0)
rewards = np.concatenate(rewards, axis=0)
penalties = np.concatenate(penalties, axis=0)