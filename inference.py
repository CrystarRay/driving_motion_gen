import torch
import os
import sys
import numpy as np
from utils.forward_diffusion import GaussianDiffusion, get_named_beta_schedule
from model.toy_diffusion import DiffusionModel
sys.path.append("./model")

checkpoint_dir = "./checkpoints"
model_name = "epoch_30.pth"
checkpoint_path = os.path.join(checkpoint_dir, model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiffusionModel().to(device)
model.eval()

if os.path.exists(checkpoint_path):
    print(f"Loading pretrained model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

T = 1000
betas = get_named_beta_schedule("linear", T)
diffusion = GaussianDiffusion(betas=betas)

input_npy_path = "./data/p49_front_1_all_3D_coordinates.npy"
motion_data = np.load(input_npy_path)

T_partial = 1000   # Use first 1000 frames
T_full = 10000     # generate 10000 frames

partial_seq = torch.tensor(motion_data[:T_partial], dtype=torch.float32).unsqueeze(0).to(device)  #(1, T_partial, 17, 3)
missing_seq = torch.randn(1, T_full - T_partial, 17, 3).to(device)

# Combine partial sequence with noise
x_T = torch.cat([partial_seq, missing_seq], dim=1)  # Shape: (1, T_full, 17, 3)

# Reverse diffusion
with torch.no_grad():
    x_t = x_T
    for t in reversed(range(T)):  # T to 0
        t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
        
        # denoise
        pred_noise = model(x_t, t_tensor)
        x_t = diffusion.q_sample(x_t, t_tensor, noise=pred_noise)



generated_motion = x_t.cpu().numpy()
print(generated_motion.shape)

model_name = model_name.replace(".pth", "")
output_path = f"./inference_output/generated_motion_{model_name}.npy"
np.save(output_path, generated_motion)
print(f"Generated motion saved to {output_path}")
