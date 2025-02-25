import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from utils.forward_diffusion import GaussianDiffusion, get_named_beta_schedule
from data.MotionCoordinatesDataset import MotionCoordinatesDataset
from torch.utils.data import DataLoader

sys.path.append("./model") 
from model.toy_diffusion import DiffusionModel 

dataset = MotionCoordinatesDataset(data_root="./data/data_root", partial_ratio=0.5)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"Total number of samples in dataset: {len(dataset)}")
print(f"Total number of batches: {len(train_loader)}")

"""
#  one sample
for batch_idx, (partial_seq, full_seq) in enumerate(train_loader):
    print(f"\nBatch {batch_idx + 1}:")
    print(f"Partial sequence shape: {partial_seq.shape}")
    print(f"Full sequence shape: {full_seq.shape}")
    
    print("Partial sequence sample (first patient, first 3 joints):")
    print(partial_seq[0, :3]) 

    print("Full sequence sample (first patient, first 3 joints):")
    print(full_seq[0, :3])

    break
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

T = 1000
betas = get_named_beta_schedule("linear", T)
diffusion = GaussianDiffusion(betas=betas)
save_every = 5
num_epochs = 30
checkpoint_dir = "./model/checkpoints"

for epoch in range(num_epochs):
    for partial_seq, full_seq in train_loader:
        optimizer.zero_grad()

        partial_seq = partial_seq.to(device)
        full_seq = full_seq.to(device)

        # Sample a random timestep for each sequence in batch
        t = torch.randint(0, T, (full_seq.shape[0],), device=full_seq.device)

        # add noise
        noise = torch.randn_like(full_seq)
        x_t = diffusion.q_sample(full_seq, t, noise=noise)

        # denoise
        pred_noise = model(x_t, t, partial_seq=partial_seq)

        loss = loss_fn(pred_denoised, full_seq)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")

    # checkpoint
    if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        latest_path = os.path.join(checkpoint_dir, "latest.pth")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, checkpoint_path)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }, latest_path)

        print(f"Checkpoint saved: {checkpoint_path}")
