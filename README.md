# GenerativeAdversarialNetworks
This notebook includes a variety of Generative Adversarial Networks (GAN's).

## Image GAN models
So far the models included is:
 * [GAN](https://arxiv.org/abs/1406.2661)
 * [DCGAN](https://arxiv.org/abs/1511.06434)
 * [WGAN](https://arxiv.org/abs/1701.07875) and [WGAN-GP](https://arxiv.org/abs/1704.00028)
 * [C-GAN](https://arxiv.org/abs/1411.1784)
 * [Progressive-GAN](https://arxiv.org/abs/1710.10196)
 * [Relativistic-GAN](https://arxiv.org/abs/1807.00734)

All models are only implemented for MNIST because of the lack of computing ressources but could easily be extented to another dataset like CIFAR10 (might come later).

## Text GAN models
The included models is:
 * [LSTM-GAN](https://arxiv.org/abs/1611.04051)
 * [SA-LSTM-GAN] Self-attention applied to LSTM-GAN
 * [RelGAN](https://openreview.net/forum?id=rJedV3R5tm)
 * [Mem-GAN] A learned Memory interation with the LSTM-GAN
