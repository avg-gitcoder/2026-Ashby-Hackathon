# 2026-Ashby-Hackathon

## Installation

Start with [INSTALL.md](INSTALL.md) for environment-specific requirements, Delta setup,
notebook kernel dependencies, model downloads/access, and recovery from missing packages.
PathGen must use a separate environment from MUSK/MedGemma.

## MUSK morphology classification

Five independent three-class predictions (-1 / 0 / +1) from histopathology image patches.

- [Server setup](musk/SERVER_GUIDE.md)
- [CSV fine-tuning instructions and data limitations](musk/FINETUNING.md)
- [Training entry point](musk/train_morphology.py)
- [Model head](musk/morphology_head.py)

The supplied annotation CSV is included; images and model weights must be accessed separately.

- [Slurm inference instructions](musk/INFERENCE.md)

- [PathGen–MedGemma–MUSK integrated report](connect/README.md)
