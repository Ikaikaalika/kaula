# Project Proposal

## Title
Efficient World Modeling with Joint Embeddings, State-Space Dynamics, and Latent Diffusion

## Motivation
World models aim to learn compact internal representations of how environments evolve over time. These models are increasingly important for video understanding, robotics, planning, model-based reinforcement learning, and agentic AI. Transformers have been widely used for temporal modeling, but they can become expensive as sequence length grows. State-space models offer a more efficient alternative for long-range sequence processing, while diffusion models offer a principled way to represent uncertainty and multimodality in future prediction. This project combines these strengths into a single joint-embedding world model.

## Research Question
Can a joint-embedding state-space world model with latent diffusion achieve competitive predictive performance while improving efficiency relative to transformer-based world-model baselines?

## Hypotheses
1. SSM-based dynamics can match or outperform transformer baselines on one-step and multi-step latent prediction at lower memory cost.
2. Adding latent diffusion improves future-sample diversity and best-of-k prediction quality relative to deterministic latent prediction.
3. A joint embedding space across observations, actions, and optional language improves representation consistency and downstream predictive robustness.

## Method
The model contains:
- an observation encoder
- a joint latent space
- an SSM dynamics core
- a latent diffusion head
- a decoder for future observation prediction

Core equations:
- z_t = E(x_t)
- h_{t+1}, ẑ_{t+1} = f_SSM(h_t, z_t, a_t)
- z*_{t+1} = f_diffusion(ẑ_{t+1}, h_{t+1}, τ, ε)
- x̂_{t+1} = D(z*_{t+1})

## Datasets
Phase 1: synthetic moving-shapes data  
Phase 2: EPIC-KITCHENS, Ego4D, or Something-Something V2  
Phase 3: DROID, BridgeData V2, or MineRL

## Baselines
- deterministic latent predictor
- transformer world model
- SSM-only world model
- full SSM + latent diffusion world model

## Metrics
- latent MSE
- cosine similarity
- reconstruction loss
- rollout error over horizon
- memory usage
- training throughput
- inference throughput

## Deliverables
- polished GitHub repo
- benchmark tables
- qualitative rollout figures
- technical report
- demo video / presentation
