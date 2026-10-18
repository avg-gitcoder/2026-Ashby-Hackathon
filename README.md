# 2026-Ashby-Hackathon

## MUSK morphology classification

Five independent three-class predictions (-1 / 0 / +1) from histopathology image patches.

- [Server setup](musk/SERVER_GUIDE.md)
- [CSV fine-tuning instructions and data limitations](musk/FINETUNING.md)
- [Training entry point](musk/train_morphology.py)
- [Model head](musk/morphology_head.py)

The supplied annotation CSV is included; images and model weights must be accessed separately.

- [Slurm inference instructions](musk/INFERENCE.md)
