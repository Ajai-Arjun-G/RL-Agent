import numpy as np
from ImprovedPolicy import PathTraceEnv  # Assuming your environment class is in this module

if __name__ == '__main__':
    # Initialize the environment
    env = PathTraceEnv()
    obs, _ = env.reset(seed=48)  # Set seed for reproducibility
    
    # Run the environment for a fixed number of steps to visualize the road scrolling
    max_steps = 2000  # Maximum steps to visualize
    step = 0
    
    while step < max_steps:
        # Scroll the road without stepping the agent
        env._scroll_road()
        
        # Render the environment (should show only the road)
        env.render()
        
        step += 1
    
    print("Total steps taken:", step)
    print("Reached maximum steps for visualization.")
    
    # Close the environment
    env.close()