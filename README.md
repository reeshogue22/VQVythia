# VQVythia (2024)

Video frame prediction in a discrete latent space.

- `vqvae.py`, `train_vqvae.py`: a VQ-VAE with an adversarial discriminator that maps 256×256
  frames to a grid of codebook indices.
- `net.py`, `train.py`: a transformer over (height, width, time) code grids whose attention
  projections are 3-D convolutions; it predicts the next frame's codes.

A February 2024 small-scale experiment; no trained weights or results are included. A
possible next step noted at the time: an idempotent-generative-network style refinement
objective ([arXiv:2311.01462](https://arxiv.org/abs/2311.01462)).

MIT licence. `ranger.py` is a vendored copy of [Ranger21](https://github.com/lessw2020/Ranger21) by Less
Wright and contributors (Apache-2.0, `LICENSE.ranger21`), with the class renamed `Ranger`.
