---
layout: post
title:  "Generative Modelling Using VQ-VAES"
date:   2023-03-05 00:00:00 +0100
categories: machine-learning
featured_img: /assets/generative-modelling-using-vq-vaes/reverse-discrete-diffusion.gif
show_excerpts: True
excerpt: "Training transformers on vq-vae latent vectors for generative modelling."
---

<sup>__note__: Some of the relevant code for this post is [here](https://github.com/mauicv/vaegan) *(Most of the training scripts are in google colabs, the above repo mostly contains model definitions and utility scripts)*</sup>

___

## Introduction

This post continuous on from this post on [vq-vaes]({% post_url 2023-03-04-vq-vaes-and-perceptual-losses %}). In it I talk about what a VQ-VAE is and how it can be trained. In this post I'll discuss how to use the latent vectors from a VQ-VAE to train a transformer for generative modelling. In particular i'll mention two main approaches the first was introduced in the paper [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) and the second is somewhat exploratory, but roughly adapts [diffusion models](https://arxiv.org/abs/2006.11239) for discrete latent spaces using transformers and inspired by the famous [stable diffusions paper](https://arxiv.org/abs/2112.10752).


## Generative Modelling using Sequential Transformers

[Transformers](https://arxiv.org/abs/1706.03762) are a class of generative model that take a sequence of discrete values and predict the next token in the sequence. They've proven to be very effective at a variety of tasks including language modelling, image classification and image captioning. I won't go into the details of how transformers work here, but if you're interested in learning more I'd recommend the [illustrated transformer](http://jalammar.github.io/illustrated-transformer/) blog post.

One idea you might have is to treat a image as a sequence of pixels and find some way to discretize there continuous values. Perhaps you could choose bins and depending on the value of each channel you'd assign it a particular token. This as been done and does work but doesn't scale well to large images. This is due to the attention mechanism in the transformer which requires to compare every token in the sequence with every other token in the sequence. This means that for a image of size `(128, 128, 3)` we'd have to compare `128 * 128` tokens with each other. This is a lot of computation and is not feasible for large images. However, using our VQ-VAEs we can get around this problem.

Previously I've demonstrated how VQ-VAEs can be trained. In doing so we've obtained a model that can be used to encode images into a discrete latent space. As an example you might have a image of shape `(128, 128, 3)` and using the VQ-VAE we can encode it into a latent space of shape `(16 * 16, 1)` where the final dimension in this vector is an integer value corresponding to one of the code-book vectors. Assuming the code-book size is `256` then we've gone from `128*128*3 float32` tensors to `16 * 16 int8` tensors. This means two things firstly the data now has a discrete representation and secondly this representation is now much smaller than the original instance. Both of these properties are desirable for training transformers as now when we train the transformer the attention mechanism will only have to compare `16 * 16` tokens with each other. This is much more manageable and once we've generated a new sequence of tokens we can decode it back into the large image space.

Training transformers is remarkably simple. The main complexity is in preprocessing the training data which requires converting each image into its discrete latent space representation. The transformer takes the sequence of tokens, $$(s_0, s_2, ..., s_n)$$ of shape `(n, 1)` and outputs a sequence of probability vectors of shape `(n, 256)`. We then use the cross entropy loss to compare the output probability vectors with the ground truth tokens. Importantly the $$s_i^{th}$$ token is used as the target for the $$p_{i-1}^{th}$$ probability vector.

Once we've trained the transformer we can generate new images by feeding the transformer a initial random token and sequentially predicting the next token in the sequence until we've generated all `16 * 16` tokens. We can then decode the sequence of tokens back into the original image space. I spend a while trying to figure out why my transformer seem to train well but not generate good images only to discover that I had an off by one error in the code that sampled the next token in the sequence 🤦! Once I fixed this the images started to look much better

## Results

In each of the following the generated images are on the left and the reconstructions are on the right. The first set shares none of the original tokens with the reconstructed images. So here we're just comparing the quality of the generated images to the reconstructions.

![generated images 1](/assets/generative-modelling-using-vq-vaes/generations-1.png)

In the second set of images i've generated the images on the left using the first `4*16` tokens from the original discrete encoding. Thus as you should be able to see the top portion of each image matches the top portion of the reconstruction.

![generated images 2](/assets/generative-modelling-using-vq-vaes/generations-2.png)

In the second set of images i've generated the images on the left using the first `8*16` tokens from the original discrete encoding.

![generated images 3](/assets/generative-modelling-using-vq-vaes/generations-3.png)


## Generative Modelling using Discrete Diffusion Transformers

[Diffusion models](https://arxiv.org/abs/2006.11239) are a class of generative model that are trained to gradually reverse a diffusion process applied to data. If we have a data point $$x_0$$, then a diffusion process samples and perturbs this point with a sequence of random values drawn from a Gaussian distribution. $$x_1 = x_0 + \xi_0$$, $$x_2 = x_1 + \xi_1$$, $$x_3 = x_2 + \xi_2$$ and so on. To train a diffusion model we train a model that tries to predict the noise value $$\xi_i$$ given a specific $$x_i$$ in the diffusion process. So we want it to minimize the mean squared error between the true noise vector and the output of the model $$\text{mse}(g(x_{i}), \xi_{i-1})$$.

This means if we have a trained diffusion model we can take a noise sample and sequentially remove noise until its less noisy. In particular if we choose an initial sample thats sampled from a Gaussian distribution then we can gradually remove noise until we get obtain something that lies within the data distribution.

This idea has been applied in image spaces most notably by OpenAI and also in latent spaces by the authors of the [stable diffusion paper](https://arxiv.org/abs/2112.10752). In the stable diffusion paper they train a VQ-VAE and then use the latent vectors as the input to a diffusion model. Thus they do so in the continuous representation prior to the discrete encoding. 

I wanted to try the above but using the discrete latent space instead of the continuous one. In order to do so I use the same transformer architecture as in the transformer case. To train the discrete diffusion model we first need to generate the training data. To do this we take the encoded discrete representation of an image, $$x$$, and randomly perturb a random percentage of the tokens in the sequence to get $$x_a$$. Next we further perturb $$x_a$$ to obtain $$x_b$$ by randomly changing a fixed percentage of the tokens this time. In my experiments I perturbed 10 percent between $$x_a$$ and $$x_b$$. We then train the diffusion model to minimize the loss $$mse(x_a, g(x_b))$$. Sampling from this model is very similar to the continuous case except that we're directly sampling the next value in the reversed diffusion process rather than the noise vector. Thus we sample a random sequence of tokens and continuously apply the transformer until it converges.

## Results

Here we show the sequence of images generated in the reverse diffusion process. In this first example we use a lower temperature when sampling the tokens in the next sequence. This results in a slower process. 

![reverse-discrete-diffusion-1](/assets/generative-modelling-using-vq-vaes/reverse-diffusion-1.png)

In the second example we use a higher temperature when sampling the tokens in the next image in the sequence.

![reverse-discrete-diffusion-1](/assets/generative-modelling-using-vq-vaes/reverse-diffusion-2.png)


![reverse-discrete-diffusion](/assets/generative-modelling-using-vq-vaes/reverse-discrete-diffusion.gif)


## Conclusion

In comparison the transformer results are definitely better although I think some improvements to the process of training the discrete case are possible. What surprised me though, is how simple the diffusion process was to set up. I think it was maybe a two or three line change in the training script. 

I think many of the nice things about diffusion models, such as guiding the reverse process to generate a specific class using pre-trained classifiers, wouldn't be possible using this approach however. This is due to running the process in the discrete space rather than the continuous one making it harder to adjust the reverse diffusion process. 

One potential benefit is that It does mean that we can use the transformer architecture instead of the UNet that's used in the continuous case. Transformers have been proven to be very performative in other domains. However, note that the way we train the loss for the for the discrete diffusion process mostly doesn't take advantage of the multiple input output pairs available in sequential data in the same way that the conventional transformer does.
