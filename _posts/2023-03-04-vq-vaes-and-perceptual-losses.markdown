---
layout: post
title:  "Vector quantized VAEs and perceptual losses"
date:   2023-03-04 00:00:00 +0100
categories: machine-learning
featured_img: /assets/vector-quantized-vaes-and-perceptual-losses/example-1.png
show_excerpts: True
excerpt: Using learned perceptual losses to train Vector quantized variational autoencoders
---

<sup>__note__: Some of the relevant code for this post is [here](https://github.com/mauicv/vaegan) *(Most of the training scripts are in google colabs, the above repo mostly contains model definitions and utility scripts)*</sup>

___

## Introduction

This post follows on from my previous post on [discriminator-perceptual-loss-for-vaes]({% post_url 2023-03-03-discriminator-perceptual-loss-for-variational-autoencoders %}). In that post I introduced the idea of using a discriminator to learn a perceptual loss for a VAE. In this post we'll focus on improving the variational autoencoder itself. Specifically, we'll be using a vector quantized VAE (VQ-VAE) to improve the quality of the latent space. VQ-VAEs are introduced in the paper [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937).

## VQ-VAEs

The idea behind VQ-VAEs is to use a discrete latent space instead of a continuous one. This is done by using a vector quantization layer to map the continuous latent space to a discrete latent space. Its not immediately obvious why this would be useful but often datasets contain information that is discrete in nature. For example, a person may or may not be wearing glasses in an image but they can't be wearing 0.5 glasses. This is a discrete property of the image. In this case, a continuous latent space is not a good fit for the data. It turns out that using a discrete latent space significantly improves the quality of the the reconstructions.


The vector quantization layer looks like an embedding layer as used in nlp. We have two parameters the number of embeddings and the embedding size. When we're encoding an image we first apply the encoder network to the image. This typically looks like a series of convolutional layers. Each of these reduce the spatial dimensions of the image while increasing the number of channels. For example, a sequence of such layers can map an image of say size `(3, 128, 128)` to a tensor of size `(512, 32, 32)`. In this case we'd choose a VQ layer with an embedding size of `512`. We'd also choose some number of embedding vectors to make up the encoding dictionary, for this example assume `256`. Thus the embedding layer has a code-book of size `(256, 512)`. When we apply the discrete encoding to the `(512, 32, 32)` tensor we map each of the `32x32` vectors to one of the `256` code-book vectors. Specifically the code-book vector that is [closest](https://github.com/mauicv/vaegan/blob/c07da10804fb4b90a099d3c546efbecd93bda1fa/duct/model/latent_spaces/discrete.py#L60).

Once this is done we've taken an image of shape `(3, 128, 128)` and compressed it down into `(32, 32, 1)` where the final dimension corresponds to the index of the code-book vector that was used to encode the vector. To reverse this process we map each index to the corresponding code-book vector and then apply the decoder network to the result. The decoder network is typically a sequence of deconvolutional layers that reverse the process of the encoder network. The result is a tensor of shape `(3, 128, 128)` which is the reconstructed image. Note that within this process the vq latent space serves to snap the continuous latent space output by the convolutional encoder to a set of discrete vectors of the same size.

This process has an issue which is that the snapping of each vector onto the nearest code-book vector is not differentiable which is a problem as backpropagation requires differentiability to train the model. To get around this we use a [straight through estimator](https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0) to make the gradient flow through the discrete layer. The way this works is to just pass the gradient from the discretized vector through the non-differentiable `argmax` operation to the output of the encoder. When doing this in pytorch there is a very simple [trick](https://github.com/mauicv/vaegan/blob/c07da10804fb4b90a099d3c546efbecd93bda1fa/duct/model/latent_spaces/discrete.py#L84) we can use:

```py
inputs  # (32 * 32, 512) dim vector output by encoder conv stack, reshaped and reordered with channel first
quantized  # (32 * 32, 512) dim vector of inputs snapped to nearest code-book vectors
quantized = inputs + (quantized - inputs).detach() # Straight through estimator
```

Doing the above means that the values of the quantized vectors are unchanged but allows the gradient to pass from the quantized tensor to the inputs tensor. Note that this means the embeddings are not updated during training. This is fine as we'll add terms in the loss to do so directly.

## Training

The loss for the vq-vae as defined in the [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) paper is:

$$
L = \log p(x|z_q(x)) + ||sg[z_e(x)] - e||^2 + \beta ||z_e(x) - sg[e]||^2
$$

where:

1. $$z_e(x)$$ is the output of the encoder network
2. $$e$$ is the embedding vector
3. $$sg$$ means stop gradients
4. $$z_q(x)$$ is the output of the quantization layer
5. $$\beta$$ should be less than $$1$$ usually around $$0.25$$. 

The first term is the typical reconstruction loss in VAEs and as mentioned in the paper:

> <sup>Due to the straight-through gradient estimation of mapping from $$z_e(x)$$ to $$z_q(x)$$, the embeddings $$e_i$$ receive no gradients from the reconstruction loss $$\log p(z\|z_q(x))$$. *(from [here](https://arxiv.org/pdf/1711.00937.pdf))*</sup>

This means the embedding vectors aren't learnt unless we add the second term $$\|sg[z_e(x)] - e\|^2$$ which is known as the code-book loss. This acts to encourage the code-book (embedding) vectors to be close to the vectors that are being output by the encoder. The `sg` operator (stop gradients) here means that the model doesn't change the encoder outputs while doing so.

Finally, the third term encourages the encoder to output vectors that are close to the code-book vectors. This is added because if the encoder outputs aren't pushed towards the code-book vectors they can instead grow arbitrarily large.

> <sup>Since the volume of the embedding space is dimensionless, it can grow arbitrarily if the embeddings $$e_i$$ do not train as fast as the encoder parameters. To make sure the encoder commits to an embedding and its output does not grow, we add a commitment loss *(from [here](https://arxiv.org/pdf/1711.00937.pdf))*</sup>

So together these two terms mean that the encoder output vectors and the embedding vectors are both pushed towards each other but the encoder output vectors more slowly than the embedding vectors.

Finally as mentioned in [discriminator-perceptual-loss-for-vaes]({% post_url 2023-03-03-discriminator-perceptual-loss-for-variational-autoencoders %}) at the same time as training the vq-vae we also train a discriminator and replace the typical reconstruction loss with a perceptual loss. This gives a much higher reconstruction quality.

### Code-Book collapse

In practice, VQ-VAES suffer from something known as code-book collapse. This is when the encoder outputs vectors that all snap onto a small subset of the code-book and the rest of the code-book is unused. This is because no term in the loss encourages the encoder to use the entire set of code-book vectors. As a result the decoder has much less information to work with when reconstructing the images and the quality of the reconstructions suffers. The solution to this problem is to keep track of the use counts of each vector in the code-book and if a vectors use falls bellow a certain threshold then we reassign the code-book vector to an encoder output vector. Doing so forces the encoder to use the new code-book vector. This is the solution used in the [OpenAI jukebox paper](https://arxiv.org/pdf/2005.00341.pdf) and seems to work really well.

## Results

| ![C-1](/assets/vector-quantized-vaes-and-perceptual-losses/C-1.png) |
| ![C-2](/assets/vector-quantized-vaes-and-perceptual-losses/C-2.png) |
| ![D-1](/assets/vector-quantized-vaes-and-perceptual-losses/D-1.png) |
| ![D-2](/assets/vector-quantized-vaes-and-perceptual-losses/D-2.png) |

| ![real image 1](/assets/vector-quantized-vaes-and-perceptual-losses/A-real.png) | ![fake image 1](/assets/vector-quantized-vaes-and-perceptual-losses/A-recon.png) |
| ![real image 2](/assets/vector-quantized-vaes-and-perceptual-losses/B-real.png) | ![fake image 2](/assets/vector-quantized-vaes-and-perceptual-losses/B-recon.png) |


## Next Steps

In the [next]({% post_url 2023-03-05-generative-modelling-using-vq-vaes %}) post I'll talk a little about using transformers to generate the discrete latent codes for the vq-vae model and in doing so turn them into generative models.