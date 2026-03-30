import mlx.core as mx
import mlx.nn as nn
from .data import patchify_frames, unpatchify_tokens

class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.gelu(layer(x))
        return self.layers[-1](x)

class PatchEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels
        self.patch_size = cfg.patch_size
        self.proj = nn.Linear(patch_dim, cfg.latent_dim)
        self.norm = nn.LayerNorm(cfg.latent_dim)
    def __call__(self, frames):
        patches = patchify_frames(frames, self.patch_size)
        return self.norm(self.proj(patches))

class PatchDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.channels = cfg.channels
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels
        self.proj = nn.Linear(cfg.latent_dim, patch_dim)
    def __call__(self, token_latents):
        return unpatchify_tokens(self.proj(token_latents), self.patch_size, self.image_size, self.channels)

class SimpleSSMCell(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(cfg.latent_dim + cfg.action_dim + cfg.ssm_hidden_dim, cfg.hidden_dim)
        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.ssm_hidden_dim)
        self.update_proj = nn.Linear(cfg.hidden_dim, cfg.ssm_hidden_dim)
        self.out_proj = nn.Linear(cfg.ssm_hidden_dim, cfg.latent_dim)
        self.norm = nn.LayerNorm(cfg.ssm_hidden_dim)
    def __call__(self, z_t, a_t, h_t):
        x = mx.concatenate([z_t, a_t, h_t], axis=-1)
        hidden = nn.gelu(self.in_proj(x))
        gate = nn.sigmoid(self.gate_proj(hidden))
        update = nn.tanh(self.update_proj(hidden))
        h_next = self.norm(gate * h_t + (1.0 - gate) * update)
        return self.out_proj(h_next), h_next

class DynamicsModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cell = SimpleSSMCell(cfg)
    def init_state(self, batch_size):
        return mx.zeros((batch_size, self.cfg.ssm_hidden_dim))
    def __call__(self, z_seq, action_seq):
        b, t, d = z_seq.shape
        h = self.init_state(b)
        preds, states = [], []
        for i in range(t - 1):
            z_pred, h = self.cell(z_seq[:, i], action_seq[:, i], h)
            preds.append(z_pred); states.append(h)
        return mx.stack(preds, axis=1), mx.stack(states, axis=1)

class DiffusionDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.time_embed = MLP([1, cfg.diffusion_hidden_dim, cfg.latent_dim])
        self.net = MLP([cfg.latent_dim * 3, cfg.diffusion_hidden_dim, cfg.diffusion_hidden_dim, cfg.latent_dim])
    def __call__(self, noisy_latent, t_scalar, context_latent, hidden_state_latent):
        if t_scalar.ndim == 1:
            t_scalar = t_scalar[:, None]
        t_embed = self.time_embed(t_scalar)
        return self.net(mx.concatenate([noisy_latent + t_embed, context_latent, hidden_state_latent], axis=-1))

class JointEmbeddingWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(cfg)
        self.token_pool = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.dynamics = DynamicsModel(cfg)
        self.hidden_to_latent = nn.Linear(cfg.ssm_hidden_dim, cfg.latent_dim)
        self.denoiser = DiffusionDenoiser(cfg)
        self.latent_to_tokens = MLP([cfg.latent_dim, cfg.hidden_dim, cfg.latent_dim])
        self.decoder = PatchDecoder(cfg)
    def encode_frames(self, frames):
        token_latents = self.encoder(frames)
        pooled = mx.mean(self.token_pool(token_latents), axis=2)
        return token_latents, pooled
    def decode_latent(self, latent):
        num_patches = (self.cfg.image_size // self.cfg.patch_size) ** 2
        token_latent = mx.broadcast_to(self.latent_to_tokens(latent)[:, None, :], (latent.shape[0], num_patches, self.cfg.latent_dim))
        return self.decoder(token_latent)
    def __call__(self, frames, actions, diffusion_t=None, noise=None, alpha_bars=None):
        token_latents, pooled = self.encode_frames(frames)
        z_preds, h_states = self.dynamics(pooled, actions)
        out = {"token_latents": token_latents, "pooled_latents": pooled, "z_preds": z_preds, "h_states": h_states, "next_frame_pred": self.decode_latent(z_preds[:, -1])}
        if diffusion_t is not None and noise is not None and alpha_bars is not None:
            target = pooled[:, -1]
            a_bar = alpha_bars[diffusion_t].reshape((-1,1))
            noisy = mx.sqrt(a_bar) * target + mx.sqrt(1.0 - a_bar) * noise
            hidden_as_latent = self.hidden_to_latent(h_states[:, -1])
            t_float = diffusion_t.astype(mx.float32) / float(alpha_bars.shape[0] - 1)
            out["eps_hat"] = self.denoiser(noisy, t_float, z_preds[:, -1], hidden_as_latent)
            out["noisy_latent"] = noisy
        return out
