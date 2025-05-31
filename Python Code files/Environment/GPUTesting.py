import torch
import os
print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.device_count())
print("Device Name:", torch.cuda.get_device_name(0))

print("Current working directory:", os.getcwd())
MODEL_PATH = "best_model_image_cnn_NewRewards"
if os.path.exists(f"{MODEL_PATH}.zip"):
    print(f"Model file found at: {os.path.abspath(f'{MODEL_PATH}.zip')}")
else:
    print(f"Model file not found at: {os.path.abspath(f'{MODEL_PATH}.zip')}")