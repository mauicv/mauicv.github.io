__note__: Some of the relevant code for this post is [here](https://github.com/mauicv/transformers)

___

## Introduction


In this post I'll look at mixture of expert architectures for transformers. In particular instead of the typical routing mechanism utilized in Mixture of expert architectures, we'll be exploring an implementation of expert choice routing introduced by [Zhou, et al](https://arxiv.org/pdf/2202.09368.pdf). We'll discuss the trade offs and then illustrate the performance on a shakespeare dataset.

## Mixture of Experts

The general idea behind mixture of experts is to have a set of distinct models (experts) that specialize in different tasks. We then route the input or the task to the model that is best suited to handle it. This can be applied to transformers in a number of ways. In our case instead of a single feed forward network in each transformer layer we're going to have a set of FFNs, these will be the experts. When the model is applied to a sequence of tokens, some routing mechanism will chose how to route tokens to experts. _Note this idea has also been applied to the [attention layers](https://arxiv.org/abs/2312.07987)_. 

![](/posts/moe-expert-choice/moe-token-choice.png)

<sup>Taken from [Mixture-of-Experts with Expert Choice Routing, Zhou, et al](https://arxiv.org/pdf/2202.09368.pdf)</sup>

Why do we do this? Well we're trying to pack as many parameters as possible into the model while keeping the runtime small. One option to scaling a transformer is to increase the matrix sizes in the FNN in each layer but this would impact runtime adversely. If instead we duplicate the FFNs and run a subset of them on the tokens we can increase the number of parameters without increasing the runtime. This means we need to choose which subset of FFNs to run on each token. This is where the routing mechanism comes in.

The simplest approach to routing is to have a routing model which is essentially just a feed forward network that takes each token as input and outputs a probability distribution over the experts. Most of the approach variation in mixture of experts comes from how we use this distribution to select the token-expert pairs. 

## Token choice routing:

The approach taken by [shazeer et al](https://arxiv.org/pdf/1701.06538.pdf) is to take the top $$k$$ values of this distribution to determine which expert to route the token to. let $R$ be the router FFN. Each token $x$ is mapped to $p=R(x)$ of shape $(p_0, p_1, ..., p_e)$ where $e$ is the number of experts. Applying top $k$ to this distribution gives us the top $k$ experts for each token. Denote $k_i$ as the index of the $k$th largest value in $p$. Then the token $x$ is routed to expert $k_i$. We sum together the expert outputs to get the final output of the layer. The output of the experts is given by:

$$x\leftarrow \sum_{i=0}^k p_{k_i} E_{k_i}(x)$$

Note that we multiply expert output by the probability of routing to that expert. By doing so we make the routing mechanism differentiable which allows us to train the router and experts jointly.

The problem with the above is that the tokens choose the experts and as a result we don't guarantee uniform use of experts. This can lead to some experts being under utilized. To address this, [Fedus, Zoph, et al](https://arxiv.org/pdf/2101.03961.pdf) propose adding an auxiliary loss to the router. The loss is defined as:

$$l_{aux} = \alpha e \sum_{i=1}^e f_i P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is a uniform distribution over the experts. This loss encourages the router to route tokens uniformly to the experts.

Instead of taking the above approach we're going to look at expert choice routing.

## Expert Choice Routing:

Introduced by [Zhou, et al](https://arxiv.org/pdf/2202.09368.pdf), expert choice routing is so named because it lets the experts choose the tokens. The routing mechanism is very similar to token choice except that each expert selects the tokens instead of the tokens selecting experts. 

![](/posts/moe-expert-choice/moe-expert-choice.png)

<sup>Taken from [Mixture-of-Experts with Expert Choice Routing, Zhou, et al](https://arxiv.org/pdf/2202.09368.pdf)</sup>

This has a couple of immediate implications. 

Firstly, It allows use to guarantee uniform use of experts, in particular it ensures that each expert sees the same number of tokens on each forward pass. 

Secondly, It means not all tokens will be routed to an expert. Some will just be passed through the residual connection to the next layer. At first this might seem like an issue but its known that early layers in transformers [will often](https://browse.arxiv.org/pdf/2309.03883.pdf) [correctly predict](https://arxiv.org/abs/2203.14680) the next token before later layers. The original transformer architecture is forced to use the same amount of compute for each token whereas expert choice allows the model to allocate compute to the tokens that need it most.

Because the experts are now choosing the tokens we also need to change the number of tokens chosen to account for different numbers of tokens passed to the model. Because of this the authors also define the capacity, $c$, which denotes on average how many experts are utilized by a token. We can compute $k$ for the `topk` operation using $k = \frac{c*n}{e}$ where $n$ is the number of tokens in the sequence and batch and $e$ is the number of experts.

## Comparisons:

In order to evaluate the expert choice routing implementation above I've selected the shakespeare character dataset, mostly motivated by [nanogpt](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#quick-start). This is a dataset of the entire works of shakespeare split by letter rather than word. Its a very small dataset that can be used to train a small transformer on a single GPU in a few minutes. This makes it ideal for testing. Note that, i'm definitely overfitting to this dataset but that's not the point of this post.

Expert choice models are meant for scaling transformers. This means there are certain regimes in which there performance becomes better. In particular, there is an associated time cost to loading and unloading each of the expert models on the GPU. This means that the performance of expert choice MoE layer scales with the batch size and sequence length.

Another more efficient way to improve MoE models is to use multiple GPUs. Doing so means we can place an expert on each GPU and avoid the time cost of loading and unloading the expert models. As well as this it means we can compute the expert outputs in parallel. In this post we're restricting ourselves to a single GPU however.

To obtain proof of concept for expert choice over token choice or standard transformer architectures we require to show a model with more parameters demonstrating faster convergence. We don't need each training step to be faster, just that the model converges faster. However, I managed to find a configuration that does both.

![](/posts/moe-expert-choice/loss-comparison.png)

The above illustrates the loss during training obtained by each model where i've plotted the x-axis in time rather than training steps so that you can see the Expert choice model converges faster and completes training quicker than both the token choice and standard transformer models. In each case the mixture models are larger and have more parameters than the standard transformer (specifically, basic transformer ~ 75 million parameters, token choice MoE & expert choice MoE ~ 126 million parameters). Note that, in order to see the benefits of expert choice routing we need to use a larger model. This means we are over fitting to the dataset.

If you'd like to experiment with the code yourself you can take a look at the [model repo](https://github.com/mauicv/transformers), or the following notebooks gists for each training run:

1. [Standard Transformer](https://gist.github.com/mauicv/d06d7c38bba222faff8c6b55f80e03d0)
1. [Token Choice MoE](https://gist.github.com/mauicv/75b8a40edafc96c0f9d6a84b16f3c708)
1. [Expert Choice MoE](https://gist.github.com/mauicv/cc2cc776f524d0144ba9ee0e53436bd6)

## Conclusion:

In this post we've looked at the mixture of experts layer and how it can be used to scale transformers. We've also looked at the expert choice routing mechanism and how it can be used to improve the performance of mixture of experts layers. Finally, we've looked at some results from training a small transformer on a single GPU and shown that expert choice routing can improve convergence speed and training time.