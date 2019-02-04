#!/bin/bash
tensorboard --logdir="RNN-GAN/pretrain" --port=5998 --window_title="RNNGAN pretrain" &
tensorboard --logdir="RNN-GAN/train" --port=5999 --window_title="RNNGAN train" &
tensorboard --logdir="SA-RNN-GAN/pretrain" --port=6001 --window_title="SARNNGAN pretrain" &
tensorboard --logdir="SA-RNN-GAN/train" --port=6002 --window_title="SARNNGAN train" &
tensorboard --logdir="MEM-GAN/pretrain" --port=6003 --window_title="MEMGAN pretrain" &
tensorboard --logdir="MEM-GAN/train" --port=6004 --window_title="MEMGAN train" &
tensorboard --logdir="RelGAN/pretrain" --port=6005 --window_title="RelGAN pretrain" &
tensorboard --logdir="RelGAN/train" --port=6006 --window_title="RelGAN train" &
tensorboard --logdir="Transformer-GAN/pretrain" --port=6007 --window_title="TransformerGAN pretrain" &
tensorboard --logdir="Transformer-GAN/train" --port=6008 --window_title="TransformerGAN train" &
