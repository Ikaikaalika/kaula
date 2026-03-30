# Model Architecture Diagrams

## High-level architecture

```mermaid
flowchart TD
    A[Observations x_t\nvideo frames / images / sensors / text] --> B[Modality Encoders]
    B --> C[Joint Embedding Space z_t]
    D[Actions a_t] --> E[Action Encoder]
    E --> F[Action Embedding]
    C --> G[SSM Dynamics Core]
    F --> G
    G --> H[Predicted Future Latent z_hat_t+1]
    G --> I[Hidden World State h_t+1]

    H --> J[Latent Diffusion Denoiser]
    I --> J
    K[Noise + timestep embedding] --> J
    J --> L[Refined / sampled future latent z_t+1*]

    L --> M[Decoder]
    M --> N[Predicted future frame / observation]

    L --> O[Reward / Value / Policy Heads]
    O --> P[Planning / Control / RL]

    C --> Q[Contrastive / JEPA-style loss]
    H --> Q
    L --> R[Rollout / imagination loop]
    R --> G
```

## Training-time internal view

```mermaid
flowchart LR
    A[x_1 ... x_t] --> B[Encoder]
    B --> C[z_1 ... z_t]

    D[a_1 ... a_t-1] --> E[Action Encoder]
    E --> F[e_1 ... e_t-1]

    C --> G[SSM / recurrent world dynamics]
    F --> G

    G --> H[z_hat_2 ... z_hat_t+1]
    G --> I[h_2 ... h_t+1]

    H --> J[Diffusion head]
    I --> J
    K[noise epsilon, timestep tau] --> J

    J --> L[denoised latent prediction]

    L --> M[Decoder]
    M --> N[x_hat_t+1]

    H --> O[latent prediction loss]
    C --> O

    N --> P[reconstruction loss]
    A --> P

    J --> Q[diffusion denoising loss]
```

## Module-by-module breakdown

```mermaid
flowchart TD
    A[Input frames] --> B[Patch / CNN / ViT encoder]
    B --> C[Per-frame tokens]
    C --> D[Pooling / projection]
    D --> E[Joint latent z_t]

    F[Action input] --> G[MLP / token encoder]
    G --> H[Action embedding a_t]

    E --> I[SSM block 1]
    H --> I
    I --> J[SSM block 2]
    J --> K[SSM block N]
    K --> L[Predicted latent z_hat_t+1]
    K --> M[Hidden state h_t+1]

    L --> N[Add diffusion noise]
    N --> O[Denoiser network]
    M --> O
    P[Timestep embedding] --> O
    O --> Q[Clean / sampled latent future]

    Q --> R[Latent-to-token projection]
    R --> S[Decoder]
    S --> T[Predicted observation]
```

## Inference / imagination loop

```mermaid
flowchart LR
    A[Current observation] --> B[Encoder]
    B --> C[Current latent z_t]
    D[Candidate action a_t] --> E[SSM dynamics]
    C --> E
    E --> F[Predicted next latent]
    E --> G[Updated hidden state]

    F --> H[Diffusion imagination]
    G --> H
    H --> I[Sampled possible future]

    I --> J[Decoder or reward/value head]
    J --> K[Score future]
    K --> L[Choose action]
    L --> E
```

## Equation summary

\[
z_t = E(x_t)
\]

\[
h_{t+1}, \hat z_{t+1} = f_{\text{SSM}}(h_t, z_t, a_t)
\]

\[
z_{t+1}^{*} = f_{\text{diffusion}}(\hat z_{t+1}, h_{t+1}, \tau, \epsilon)
\]

\[
\hat x_{t+1} = D(z_{t+1}^{*})
\]
