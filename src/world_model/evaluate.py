from __future__ import annotations

from .metrics import mse, cosine_similarity
from .datasets import build_dataset_iterator

def evaluate_one_step(model, cfg, num_batches: int = 10):
    iterator = build_dataset_iterator(cfg, num_batches=num_batches)
    latent_mses = []
    latent_cos = []
    recon_losses = []

    for batch in iterator:
        out = model(batch["frames"], batch["actions"])
        target_latent = out["pooled_latents"][:, 1:]
        pred_latent = out["z_preds"]
        latent_mses.append(float(mse(pred_latent, target_latent).item()))
        latent_cos.append(float(cosine_similarity(pred_latent, target_latent).item()))
        recon_losses.append(float(mse(out["next_frame_pred"], batch["frames"][:, -1]).item()))

    return {
        "latent_mse": sum(latent_mses) / max(1, len(latent_mses)),
        "latent_cosine": sum(latent_cos) / max(1, len(latent_cos)),
        "recon_loss": sum(recon_losses) / max(1, len(recon_losses)),
    }
