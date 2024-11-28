export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=da40a6f41c89dafc58aa3d75ff667557276bc0b0

python src/train.py experiment=flood trainer.devices=1
