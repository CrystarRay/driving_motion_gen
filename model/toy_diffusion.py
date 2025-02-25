import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DiffusionModel(nn.Module):
    def __init__(self, njoints=17, nfeats=3, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1):
        """
        Diffusion Model for Motion Generation.

        Args:
            njoints (int): Number of joints.
            nfeats (int): Number of features per joint.
            latent_dim (int): Transformer hidden dimension.
            ff_size (int): Feedforward network size in Transformer.
            num_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.latent_dim = latent_dim

        self.input_feats = self.njoints * self.nfeats

        # Maps motion input to latent space
        self.input_process = InputProcess(self.input_feats, self.latent_dim)

        # Positional Encoding for time-aware embedding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=ff_size,
                                                   dropout=dropout,
                                                   activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # diffusion conditioning
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Maps back to motion representation
        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints, self.nfeats)


    def forward(self, x_t, t, partial_seq=None):

        bs, T, njoints, nfeats = x_t.shape
        x_t = x_t.reshape(bs, T, njoints * nfeats)  # Flatten motion features

        # Embed time steps
        time_emb = self.embed_timestep(t)

        # Process input
        x = self.input_process(x_t)
        # print(f"partial_seq.shape before reshape: {partial_seq.shape}")
        # print(f"Expected reshape: {[bs, T, njoints * nfeats]}")
        
        if partial_seq is not None:
            """
            T_partial = partial_seq.shape[1]
            if T_partial < T:
                print(f"Warning: Padding partial_seq from {T_partial} to {T}")
                pad_size = T - T_partial
                pad_tensor = torch.zeros((bs, pad_size, njoints, nfeats), device=partial_seq.device)
                partial_seq = torch.cat([partial_seq, pad_tensor], dim=1)  # Pad along time axis
            """
            
            # reshape
            partial_seq = partial_seq.view(bs, T, njoints * nfeats)
            cond_emb = self.input_process(partial_seq)
            x = x + cond_emb


        # Add positional encoding
        xseq = self.sequence_pos_encoder(x + time_emb)

        # Transformer Encoding
        output = self.transformer(xseq)

        # Process output back to motion format
        output = self.output_process(output)

        return output

# Positional Encoding block
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Embeds diffusion timestep into model
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, timesteps):
        timesteps = timesteps.clamp(0, 999)
        return self.time_embed(self.sequence_pos_encoder.pe[:, timesteps]).permute(1, 0, 2)


# Input Processing Layer
class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        return self.embedding(x)

# Output Processing Layer
class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.linear = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, x):
        bs, T, _ = x.shape
        x = self.linear(x)
        return x.view(bs, T, self.njoints, self.nfeats)
    
# test
"""
if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 2
    T = 100
    njoints = 17
    nfeats = 3
    latent_dim = 256
    num_layers = 8
    num_heads = 4
    dropout = 0.1

    model = DiffusionModel(njoints=njoints, nfeats=nfeats, latent_dim=latent_dim,
                           num_layers=num_layers, num_heads=num_heads, dropout=dropout)

    x_t = torch.randn(batch_size, T, njoints, nfeats)
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long)

    output = model(x_t, t)

    print("Input shape:", x_t.shape)
    print("Output shape:", output.shape)
    print("Test Passed!" if output.shape == x_t.shape else "Test Failed!")
"""