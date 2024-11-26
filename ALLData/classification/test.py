import torch

# Check the total number of GPUs detected
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs detected: {gpu_count}")

# List GPU names and IDs
for i in range(gpu_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Test if each GPU is usable
for i in range(gpu_count):
    try:
        torch.tensor([1.0], device=f'cuda:{i}')
        print(f"GPU {i} is functional.")
    except Exception as e:
        print(f"Error with GPU {i}: {e}")
