from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .config import WorldModelConfig
from .data import make_beta_schedule
from .datasets import build_dataset_iterator
from .losses import world_model_loss
from .model import JointEmbeddingWorldModel


def build_alpha_bars(cfg: WorldModelConfig):
    betas = make_beta_schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end)
    alphas = 1.0 - betas
    alpha_bars = mx.cumprod(alphas, axis=0)
    return betas, alphas, alpha_bars


def make_model_and_optimizer(cfg: WorldModelConfig):
    model = JointEmbeddingWorldModel(cfg)
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    mx.eval(model.parameters(), optimizer.state)
    return model, optimizer


def sample_diffusion_inputs(cfg: WorldModelConfig, batch_size: int, latent_dim: int):
    t = mx.array(__import__("numpy").random.randint(0, cfg.diffusion_steps, size=(batch_size,)), dtype=mx.int32)
    noise = mx.random.normal((batch_size, latent_dim))
    return t, noise


def loss_fn(model, cfg, batch, alpha_bars):
    frames = batch["frames"]
    actions = batch["actions"]
    t, noise = sample_diffusion_inputs(cfg, frames.shape[0], cfg.latent_dim)
    outputs = model(frames, actions, diffusion_t=t, noise=noise, alpha_bars=alpha_bars)
    total, logs = world_model_loss(cfg, outputs, frames, noise)
    return total, logs


def train_epoch(model, optimizer, cfg, alpha_bars, num_batches=100, dataset_iterator=None):
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, batch: loss_fn(m, cfg, batch, alpha_bars))
    history = []
    if dataset_iterator is None:
        dataset_iterator = build_dataset_iterator(cfg, num_batches=num_batches)
    for batch in dataset_iterator:
        (loss, logs), grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        history.append({k: float(v.item()) for k, v in logs.items()})
    return history
