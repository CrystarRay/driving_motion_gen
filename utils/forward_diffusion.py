import math
import numpy as np
import torch as th


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Return a beta schedule by name.
    For a linear schedule, scales betas by scale_betas.
    """
    if schedule_name == "linear":
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_bar function.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion:
    """
    A minimal implementation of forward diffusion (q(x_t | x_0)).
    """
    def __init__(self, *, betas):
        self.betas = np.array(betas, dtype=np.float64)
        self.num_timesteps = len(self.betas)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the input x_start for a given timestep t.       
        """
        if noise is None:
            noise = th.randn_like(x_start)
        sqrt_alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1D numpy array for a batch of indices and expand to broadcast_shape.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape)


"""
if __name__ == "__main__":

    batch = 1
    frames = 100 
    joints = 17
    coords = 3
    x_start = th.randn(batch, frames, joints, coords)
    
    num_timesteps = 1000
    betas = get_named_beta_schedule("linear", num_timesteps)
    diffusion = GaussianDiffusion(betas=betas)

    t = th.randint(0, num_timesteps, (batch,), device=x_start.device).long()
    x_t = diffusion.q_sample(x_start, t)
    
    print("x_start shape:", x_start.shape)
    print("t:", t)
    print("x_t shape:", x_t.shape)
"""