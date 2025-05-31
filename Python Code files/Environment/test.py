import os
from stable_baselines3 import PPO
from ImprovedPolicy import PathTraceEnv  # Assuming your environment class is in this module

if __name__ == '__main__':
    # Path to the trained model
    MODEL_PATH = "D:/RL_Project/Version 2/Version 2"
    model_file_path = f"{MODEL_PATH}.zip"

    # Check if the model exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"No model found at {os.path.abspath(model_file_path)}. Please train the model first.")

    # Load the trained model
    model = PPO.load(MODEL_PATH)

    # Initialize the environment for testing
    env = PathTraceEnv()
    obs, _ = env.reset(seed=24)  # Set seed for reproducibility
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
    
    print("Total steps:", env.total_steps)
    env.close()