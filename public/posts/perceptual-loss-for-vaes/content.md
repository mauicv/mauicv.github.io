__note__: Some of the relevant code for this post is [here](https://github.com/mauicv/vaegan) *(Most of the training scripts are in google colabs, the above repo mostly contains model definitions and utility scripts)*

___

## Introduction

Ages ago, when I first got into machine learning I set out trying to train a GAN. I wrote a blog post on the experience [here]({% post_url 2020-07-22-generative-adversarial-networks-faces %}). GANs are tricky to train because they're so unstable so I tried using [variational autoencoders instead](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/). VAEs are very stable but if you've ever trained one you'll know that they give very blurry images and struggle to capture fine detail. For example the following is typical:

![vae trained with mse](/posts/perceptual-loss-for-vaes/mse-vae.png)

The reason this happens is because typically the loss we use to train the variational autoencoder doesn't actually correspond to perceptual similarity. In my case I was using Mean squared error which is sensitive to small translations of the image. If you have an original image $$x$$ and an image $$x'$$ which corresponds to a translation of $$x$$ by a small number of pixels, the means square error between these two images is likely to be quite large. This is because it compares the images pixel by pixel.

What we'd prefer is something that extracts general features and compares these instead. So instead of saying these two pixels are the same this metric would instead first extract features and ask if the features are the same instead. We do this using two tricks, the first of which i've alluded too already. We use the idea of a discriminator from the Adversarially trained GAN models. We'll first train this discriminator to distinguish between the real images and the reconstructions. For the second trick we use something called the perceptual loss.

### Perceptual-loss

Introduced in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), perceptual loss was originally used for style transfer.

![style-transfer](/posts/perceptual-loss-for-vaes/style-transfer.png)
_fig-3_: taken from [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

The ideas is that the pre-trained classifier, in this case [VGG](https://arxiv.org/abs/1409.1556), encodes both content and style in different convolution layers. Given an image in a specific style we can extract the style by extracting the activations in these layers when passing the image through the classifier. Subsequently we can change other images to match that images style by changing them to incur similar sets of activations when again passed through the classifier.

This is really simple to do:

```py
def loss(critic, x, y, layer_inds=None):
    batch_size = x.shape[0]
    x = critic.input_conv(x)
    y = critic.input_conv(y)
    loss_sum = 0
    for ind, layer in enumerate(critic.layers):
        x = layer(x)
        y = layer(y)
        if ind in layer_inds:
            rx = x.reshape(batch_size, -1)
            ry = y.reshape(batch_size, -1)
            loss_sum = loss_sum + ((rx - ry)**2).sum(-1)
    return loss_sum
```

### Training

Instead of using VGG I trained a discriminator along side the VAE and then used this to compute the perceptual distance on real and reconstructed images. So the training look looks like:

```
- initialize autoencoder and discriminator
- for n epochs
    - for x_batch in dataset
        - recon_batch = autoencoder(x_batch)
        - update discriminator to differentiate between recon_batch and x_batch
        - compute perceptual distance between recon_batch and x_batch using discriminator
        - update autoencoder to minimize perceptual distance
```

There is also some choice of discriminator framework. In reality the plain old adversarial loss didn't give great results. Instead the two approaches that I found worked best used the approaches detailed [here (geometric GAN)](https://arxiv.org/pdf/1705.02894.pdf) and [here (wasserstein GAN)](https://arxiv.org/pdf/1701.07875.pdf).

As well as this I used every layer in the discriminator to compute the perceptual distance. In theory the higher layers contain more stylistic information and the lower layers more conceptual information, and I wanted to match for both of these things.

### Results

![reconstructions-1](/posts/perceptual-loss-for-vaes/vaegan-reconstructions.png)
_fig-1_: Reconstructions of variational autoencoder using a perceptual loss obtained from a discriminator trained using the [geometric gan loss](https://arxiv.org/pdf/1705.02894.pdf)

![reconstructions-2](/posts/perceptual-loss-for-vaes/vaegan-reconstructions-2.png) 
_fig-2_: Reconstructions of variational autoencoder using a perceptual loss obtained from a discriminator trained using a [wasserstein loss](https://arxiv.org/pdf/1701.07875.pdf)


### Conclusions

The above is still pretty unstable and requires lots of fine tuning of hyper parameters to get right. I think because the GAN is still doing a lot of the work. In the [next]({% post_url 2023-03-04-vq-vaes-and-perceptual-losses %}) post I'll discuss [vector quantized variational autoencoders](https://arxiv.org/abs/1711.00937) which significantly improve the performance of the encoder/decoder step but result in some added complexities.
