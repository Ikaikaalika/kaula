# Scaling to 70B, hardware, and RL notes

## Can this architecture become a 70B-parameter model?
Yes, architecturally. But training a dense 70B model from scratch is a multi-accelerator datacenter job, not a local Mac job.

## Practical hardware tiers
- Local Apple silicon: architecture prototyping, toy and small models, some fine-tuning and quantized inference.
- 4–8 datacenter GPUs: serious mid-scale experiments.
- 70B dense training: typically 8x H200 / 8x B200 / 8x MI300X class systems, often across multiple nodes.

## Why not just use a Mac?
A 70B model in bf16/fp16 is ~140 GB just for weights, before gradients and optimizer state. Training usually needs far more than raw weights.

## RL only?
RL-only from near-scratch is usually a poor plan for a 70B model. A better recipe is:
1. seed/self-supervised pretraining
2. world-model training
3. RL for behavior, planning, and refinement

Using a bit of seed data and then leaning heavily on RL is plausible. Pure reward-driven training from scratch is technically possible in principle but usually sample-inefficient, unstable, and extremely expensive.
