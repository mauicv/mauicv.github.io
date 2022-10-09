---
layout: post
title:  "Learning Perceptual losses to train better Variational Autoencoders"
date:   2022-10-08 00:00:00 +0100
categories: machine-learning
featured_img: /assets/discriminator-perceptual-loss-for-vaes/vaegan-training.gif
show_excerpts: True
excerpt: Using learned perceptual losses to train better variational autoencoders
---

<sup>__note__: *Relevant code for this post is [here](https://github.com/mauicv/vaegan)*</sup>

___

## Introduction

Ages ago, when I first got into machine learning I set out trying to train a GAN. I wrote a blog post on the experience [here]({% post_url 2020-07-22-generative-adversarial-networks-faces %}). GANs are tricky to train becuase they're so unstable so I tried using [variational autoencoders instead](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/). VAEs are very stable but if you've ever trained one you'll know that they give very blurry images and struggle to capture fine detail. For example the following is typical:

![vae trained with mse](/assets/discriminator-perceptual-loss-for-vaes/mse-vae.png)

The reason this happens is becuase typically the loss we use to train the variational autoencoder doesn't actually correspond to perceptual similarity. In my case I was using Mean squared error:

$$mse(x, y) = \frac{1}{N}\sum^{N}_{i,j}{||x_{i,j}-y_{i,j}||^2}$$

To see why the MSE is a bad judge of distance between two images consider an image made up of alternating columns or stripes of black and white pixels. Now conisder the same image but shift it across by one pixel. By doing so you now have a nearly identical image but where the black and white strips have changed places. See the following image for an illustration.

![striped-images](/assets/discriminator-perceptual-loss-for-vaes/striped-imgs.png)

From our perspective the two images are nearly identical becuase they're both images of stripes and the fact that one has been moved over to the left slightly doesn't mean much. If I slightly adjusted in the same way an image of someone you knew, you'd still think it was an image of that person. However because shifting the image slightly now means that the white pixils now overlay where the black pixels once where the $$\|x_{i,j} - y_{i,j}\|^2$$ term is $$1$$ for every pixel in $$x$$ and $$y$$. So these two images end up being as far away as they possibly can be in the MSE loss.

The better solution to the above problem would be if we had a loss that looked at each image and thought these are both images of horizontal stipes but one is moved slightly to the right, the small distance it's been moved should be the error. In order for such a loss to do this it would need to have a way of detecting stable features in the image and using these to determnin distance rather than the individual pixels. In the above case perhaps a loss that detects horizontal strips would be approaprate. 

Note here that the "stable" features of interest are really dependent on the dataset we're interested in and when i say "stable" i mean commonally present within the dataset. The vertical stripe feature would be useful for detecting other images of vertical stripes but less so if the dataset is made up of images of horizonal stripes. For instance if we're trying to tell that two images of a face are close or not the we'd want to detect the presents of features such as eyes and noses and hair in order to measure simialrities between memebers of this dataset.

So there are two peices of theory we need to obtain such a metric. The first is commonly known from [Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network) in the way of discriminators. In GANs we have a generator and a discriminator. The generator tries to create images that belong to a dataset by learning to fool a discriminator that's simultaneously trained to distinguish between real images fromt he dataset and images created by the generator. In order to do this the discriminator has to learn a set of features that it can use to do just this. These are the features we want to use.

The second item we need is the idea of perceptual loss(see [here](https://arxiv.org/abs/1603.08155) and [here](https://arxiv.org/abs/1508.06576)). If you have a classifier of some kind, in our case a discriminator, this classifier has encoded within its layers features that it uses to distinguish between its classes. As you feed an image through this classifier it will activate neruons corresponding to the presence of features. These activations will then be propigated to the next layers, and activations of neruons within these layers correspond to the presence of features built out of the features extracted in the first layers and so on. So inital features will just be for the presence of lines and curves, and the later features will detecto circles and patterns and then later still whole objects and things as complex as faces. This is how nerual networks can learn to detect the presence of large structures in an image. We want more than to detect if an image is a face or not however we want a metric on faces so that we can compute the loss for a variational autoencoder. Given a pair of images, we compute the difference in the activations in the classifier network that result from passing each image through it. We then take the mean squared error of this value instead of the pixelwise differences.

So we're going to train a discriminator at the same time as we train the variational autoencoder and while doing so we also use this discriminator to compute the perceptual loss distance between the real and reconstructed images. This is the approach taken in the paper [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf) and using this method gets much better results than using the mean squared error:

| ![reconstructions-1](/assets/discriminator-perceptual-loss-for-vaes/vaegan-reconstructions.png) | 
|:--:| 
| _fig-1_: Reconstructions of variational autoencoder using a perceptual loss obtained from a discriminator trained using the [geometric gan loss](https://arxiv.org/pdf/1705.02894.pdf) | 

## Choice of adversarial loss

So the training process is the same as a conventional variational autoencoder except on each iteration we now also train a discriminator to try and distinguish the reconstructed images from the ones initailly fed into the autoencoder. We then use the discriminator to compute the perceptual distance between the original images and there reconstructions. As well as this we also use the conventional generator loss from the discriminator.

There is choice here in what type of adversarial loss we use to train the discriminator. The two setups I had the most success with where [wasserstein loss](https://arxiv.org/pdf/1701.07875.pdf), and the loss from the [Geometric GAN paper](https://arxiv.org/pdf/1705.02894.pdf) (see fig-1):

| ![reconstructions-2](/assets/discriminator-perceptual-loss-for-vaes/vaegan-reconstructions-2.png) | 
|:--:| 
| _fig-2_: Reconstructions of variational autoencoder using a perceptual loss obtained from a discriminator trained using a [wasserstein loss](https://arxiv.org/pdf/1701.07875.pdf) |

I felt latter the approach worked best so I'll talk a little more about Geometric GANs.

### Geometric GANs

Consider the archetecture of the discriminator as a set of layers that extract features and map them to an $$n$$ dimensional space, we then have a classifier layer on top that maps the $$n$$ dimensional output to a single value. We can think of this as two operations, the first is the nerual network function: $$\Phi_{\theta}: \mathbb{R}^{N\times N} \rightarrow \mathbb{R}^{k}$$. And the second is the application of the final layer. Denote the final layer weights as $$\omega\in \mathbb{R}^k$$ then we can write the whole network as $$ x \rightarrow \langle \Phi_{\theta}(x), \omega \rangle$$. This looks like the formulation for a [support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine) over the features extracted by $$\mathbb{\Phi}_{\theta}$$. This motivates the use of the Hinge-loss given by:

$$L = max(0, 1 - y_i\langle \Phi_{\theta}(x_i), \omega \rangle)$$

where $$y_i$$ is $$1$$ if the data point $$x_i$$ was generated by the autoencoder and $$0$$ otherwise. This gives us the loss for the Geometric GANs discriminator:

$$
R = \mathbb{E}_{x \sim p_{x}}[max(0, 1 - D_{\phi}(x))] + \mathbb{E}_{x \sim p_{x}}[max(0, 1 + D_{\phi}(AE_{\theta}(x)))]
$$

Where $$p_{x}$$ is the data distribution and $$AE_{\theta}$$ is the variation autoencoder.
In turn the generator loss is given by:

$$
L = -\mathbb{E}_{x\sim p_{z}}[ D_\theta(g_{\theta}(z))]
$$

Intuitivly this means the discriminator is trying to push the supporting vectors towards the margin boundaries whereas the generator is trying to push the fake feature vectors towards the seperating hyperplane.

| ![hyperplane-margin](/assets/discriminator-perceptual-loss-for-vaes/geo-gan-svm.png) | 
|:--:| 
| _fig-3_: taken from [Geometric GAN paper](https://arxiv.org/pdf/1705.02894.pdf) |

### Perceptual-loss

Introduced in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), perceptual loss was originally used for style transfer:

| ![style-transfer](/assets/discriminator-perceptual-loss-for-vaes/style-transfer.png) | 
|:--:| 
| _fig-3_: taken from [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) |

The ideas is that the pretrained classifier, in this case [VGG](https://arxiv.org/abs/1409.1556), encodes both content and style in different conolutional layers. Given an image in a specific style we can extract the style by extracting the activations in these layers when passing the image through the classifier. Subseqently we can change other images to match that images style by changing them to incur similar sets of activations when again passed through the classifier.

Typically the first layers in a convolutional nerual network encode for stylistic features and later layers, content. We're interested in all of these so we use every layer of the critic when computing perceptual loss between two images. We implement the loss function as a method on the critic. We take the activations after each layer in the critic and compute the sum of squared differences between each image.

```py
def loss(self, x, y, layer_inds=None):
    batch_size = x.shape[0]
    x = self.input_conv(x)
    y = self.input_conv(y)
    sum = 0
    for ind, layer in enumerate(self.layers):
        x = layer(x)
        y = layer(y)
        if ind in layer_inds:
            rx = x.reshape(batch_size, -1)
            ry = y.reshape(batch_size, -1)
            sum = sum + ((rx - ry)**2).sum(-1)
    return sum
```

### Conclusion:

I'm pretty pleased with these results. The process can still be fairly tricky to get to work and I tried a lot of things in the process of getting to the above. Originally I was hoping for something with the stability of variational autoencoders but the quality of GANs. I'd say that while the quality is high there are still some issues with stability. 

Recently something similar to the above was done in the [stable diffusions paper](https://arxiv.org/abs/2112.10752). There they fit [diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) in the latent space of [vector quantized variational autoencoders](https://arxiv.org/abs/1711.00937) trained with [patch discriminators](https://compvis.github.io/taming-transformers/) and perceptual losses.