In the following posts we introduce a set of model free rl methods:

- __1.__ [Natural Evolutionary Strategies](#/posts/rl-nes)
- __2.__ [Q-Learning](#/posts/rl-dqn)
- __3.__ [Policy Gradients and REINFORCE](#/posts/rl-pg)
- __4.__ [DDPG and TD3](#/posts/rl-ddpg)

A key aspect of the above methods is that they don't use a model of the environment they're trying to solve. Instead they learn by directly sampling outcomes. There are two main issues with this.

- __1. Sample efficiency__: RL typically requires a large number of interactions with the environment in order to learn the correct solution. This typically makes RL for real world problems prohibitively expensive because of the sheer number of interactions with reality that you'd have to perform.
- __2. Reality gap__: The alternative to training in reality is to create a simulated environment with the intention of moving to a real world environment after training is complete. Doing so means we run into the reality gap which is the difference between the simulation and reality. This approach also suffers from scaling issues. In order to work, we're going to have to create a simulation of every environment we want the agent to solve.

A potential solution to the above is to train a world model of the environment. This might seem like we're adding more complexity to the problem and in a sense we are. Adding the world model means more computation, but less real world sampling. Typically the real world sampling is the really slow part so this trade off makes sense. We still need to sample from the real world environment in order to train the world model so why is it more sample efficient? Well training the world model falls into the realm of traditional deep learning in the sense of $X$ input data, $Y$ labels, minimize a loss function. There's no value allocation or estimating reward gradients. Because of this it needs much less samples to train a world model of an environment than it does to train an agent in directly the environment (assuming sampling from the environment is expensive). Once we have the trained environment sampling from it becomes very fast because we can make use of hardware acceleration and generate lots of rollouts in parallel.

This proxy environment also comes with the added benefit that there is no reality gap. Its trained on the real world and so will be a much better fit to the real world than a simulation created by a developer. It also has much more potential to scale as the process of creating the simulation is automated.

## Algorithm:

The rough idea of the approach involves three steps:

- __1.__ Create a representation model that compresses the size of the input states (in our case images). We use a pair of encoder and decoder CNN networks for this. In particular we can choose to either a continuous or discrete latent representation. The benefit of compressing the state into a lower dimensional representation like this is that it makes training the world model less computationally expensive/easier. This step is probably only needed if your using images as model input. Also note that [Hafner et al](https://arxiv.org/abs/1912.01603) actually use two approaches, 1 being reconstruction learning the other is contrastive learning which performs slightly worse and I won't cover here.
- __2.__ Given a sequence $S_i = (s_i, a_i, r_i)_{i=0}^{n}$ of latent state, action and reward triple - train a world model (dynamic model) which takes $(s_i, a_i, r_i)$ and predicts $(s_{i+1}, r_{i+1})$.  the latent representation and predicts the next one in a sequence. I'll talk about two different approaches here that I tested, a recurrent state space model (RSSM) or a transformer state space model (TSSM).
- __3.__ Train an agent inside the world model to take actions that maximize rollout reward. In particular because we have a differentiable world model we can train the agent to do this directly rather than estimating the reward gradients using monte carlo methods. Note that we can obtain a world model rollout and just maximize the rewards directly but doing so means we're limited to the finite time horizon of the rollout. We can also train a value function that estimate the reward beyond this time horizon and doing so facilitates more stable learning.

The theory is in some way a lot simpler than policy gradient methods. This is because we're reliant on good old deep learning techniques and don't need fancy stuff like the policy gradient theorem. The only slightly complex feature comes in step three when we train a value function to extend beyond the time horizon. 

The approach is slightly different for the RSSM and TSSM cases so I'll deal with each separately:

# Recurrent state space model (RSSM):

We'll roughly follow [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603).

We define a rollout of the environment for a given policy as the sequences $\{(o_i,a_i,r_i)\}_{i\in\mathbb{N}}$ where $o_i$ are observations (in our case images) of the environment state, $a_i$ are actions selected according to the current policy and $r_i$ is the reward emitted from the environment in that state.

## Representation Learning:

The representation model (Called the observational model) is a variational autoencoder (VAE). Theres so much content out there about VAEs that I don't want to choose any one thing to recommend but a good place to start is [this](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) post. I've also written (badly) about them [here](#/posts/perceptual-loss-for-vaes) and [here](#/posts/vqvaes-and-perceptual-losses).

In our case a VAE is two probabilistic models, an encoder $e_{\theta}(s_t| s_{t-1}, a_{t-1}, o_t)$ that compresses images, previous states and previous actions into a latent representation, and a decoder $d_{\theta}(o_t| s_{t})$ that maps from the latent representation back to the original space. The basic idea is we train it to predict its own input and by doing so we obtain a compressed representation of the input data which we can map to and from using the encoder and decoder respectively. We'll use the typical reconstruction loss to train this model.

## Dynamic Learning

In order to learn the dynamic model we'll first sample an environment rollout $\{(o_i,a_i,r_i)\}_{i\in\mathbb{N}}$ and then map the observations to latent states using the encoder to get $\{(s_i,a_i,r_i)\}_{i\in\mathbb{N}}$. We want the dynamic model to take as input the latent representation and an embedding of the action and then predict the next latent state and reward. We use a RNN [GRU cell](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html) to learn the dynamic which means there is a recurrent hidden state that is passed between each prediction in the sequence we rollout. Thus we have:

- __1__: Dynamic model: $h_t = f_{\theta}(s_{t-1}, a_{t-1}, h_{t-1})$
- __2__: Prediction model: $s_t \sim P_{\theta}(s_{t}| h_{t})$
- __3__: Reward model: $r_t \sim R_{\theta}(r_{t}| h_{t})$

### Loss functions:

We learn the above models with the following losses ($h_t = f_{\theta}(s_{t-1}, a_{t-1}, h_{t-1})$):

$$
\begin{aligned}
\mathcal{L}_{D}^t & =-\beta\mathbb{KL}\big(e_{\theta}(s_t| s_{t-1}, a_{t-1}, o_t)|| P_{\theta}(s_{t}|h_t)\big)\\

\mathcal{L}_{O}^t &  = \ln {d_{\theta}(o_{t}| h_{t})} \\ 

\mathcal{L}_{R}^t &  = \ln {R_{\theta}(r_{t}| h_{t})} \\
\end{aligned}
$$

To compute theses we need to do so in an iterative manner due to the dependency on the hidden state at each stage of the rollout. We start by encoding all of the $o_i$ and generated a zero hidden state $h_0$. We then start at $i=0$, and compute $h_1 = f_{\theta}(s_{0}, a_{0}, h_{0})$. From this we compute the distributions $R_{\theta}(r_1| h_1)$ and $P_{\theta}(s_1|h_1)$. We store $R_{\theta}(r_1|h_1)$ and $P_{\theta}(s_1|h_1)$ and sample a new state $s_1\sim P_{\theta}(s_1|h_1)$. We now have everything we need to compute $h_2 = f_{\theta}(s_{1}, a_{1}, h_{1})$. By iterating we can compute the sequence of $R_{\theta}(r_k|h_k)$ and $P_{\theta}(s_k|h_k)$, as well as the sequence of sampled $s_k$ which we can reconstruct using $d_{\theta}(s_k)$. All of these terms together give us enough to compute the three losses above.

__Note__: some environments also have a done signal for when the agent has finished or failed at the task its trying to achieve. This is simple to add an is just a extra model $D_\theta(\text{done}|s_i)$. It should be a categorical variable so the only difference is that you train it using binary cross entropy on the signal from the environment.

## Training the agent/actor

In order to train the agent we need to be able to generate a imagined rollout. To do this we start with a zero hidden state $h_0$ as well as a state sampled randomly from the environment. We first use the state to generate an action $a_0$ using the actor. We then compute the next hidden state as when training the dynamic model and use the prediction model to sample the next state - $s_1 \sim P_{\theta}(s_{1}| h_{0})$ and the next reward $r_0 \sim R_{\theta}(r_{0}| h_{0})$. We then use the actor again to obtain $a_1=\text{actor}(s_1)$ and we can compute the next hidden state and repeat. We iterate this process for a finite imagination horizon. Once we're done we can update the actor by summing the rewards and performing gradient ascent w.r.t. the actors parameters. We can do this because all the components of the world model are differentiable - with the exception of the probabilistic models. For these we use straight through estimation and it seems to work just fine.

The negative sum of the rewards is one loss function we could use - however the finite time horizon limits how far ahead the reward signal propagates. To fix this the authors also introduce a value model $V(s)$ that computes the expected future reward from the state $s$. This is the exact same value function introduced in the [q-learning](http://localhost:3000/#/posts/rl-q-learning) post and used in the [advantage actor critic](http://localhost:3000/#/posts/rl-ac) algorithm. We'll train it using TD learning.

Remember that the TD learning uses the fact that $v_{\psi}(s_t)$ should equal the current reward plus the future expected reward for $s_{t+1}$ i.e. $r_t + \gamma v_{\psi}(s_{t+1})$. Because of the many possible future trajectories stemming off from $s_{t+1}$ the $v_{\psi}(s_{t+1})$ typically has very high variance. If we have a longer rollout then we can use this to improve the variance of $v_{\psi}(s_{t+1})$ by taking into account more $r_t$. If $k$ is the number of reward steps we want to account for and $H$ is the total length of the imagined rollout then we have:

$$
V_N^k(s_t) \doteq \mathbb{E}_{q_\theta, q_\phi} \left( \sum_{n=t}^{h-1} \gamma^{n-t} r_n + \gamma^{h-t} v_{\psi}(s_{h})\right) \text{ where } h=\min(t+k, t+H)
$$

Note that we use $h=\min(t+k, t+H)$ because $k$ might be longer than the number of steps in the rollout. The formular above is the discounted sum of rewards up to the $k^{th}$ step or end of the rollout after which we use the estimate $v_{\psi}(s_{h})$.

Also note that we need the above to be an expectation because we're estimating the value from rollouts and thus the $r_n$ are samples.

Note that the $V_N^k(s_t)$ for all $k\in\{0, ..., H\}$ are all estimates of the same quantity thus we can take the n exponential-weighted average of them all and this will also be an estimate of $V(s_t)$. This is what Hafner et al do in [DreamerV1](https://arxiv.org/pdf/1912.01603) in order to balance bias and variance. The target they end up with is:

$$
V_\lambda(s_\tau) \doteq (1 - \lambda) \sum_{n=1}^{H-1} \lambda^{n-1} V_N^n(s_\tau) + \lambda^{H-1} V_N^H(s_\tau)
$$

Thus the loss functions for the actor $a_\phi$ and value function $v_{\psi}$ become

$$
\begin{aligned}
\max_\phi \mathbb{E}_{q_\theta, a_\phi} \left( \sum_{\tau=t}^{t+H} V_\lambda(s_\tau) \right)
\qquad
\min_\psi \mathbb{E}_{q_\theta, a_\phi} \left( \sum_{\tau=t}^{t+H} \frac{1}{2} \left\lVert v_\psi(s_\tau) - V_\lambda(s_\tau) \right\rVert^2 \right)
\end{aligned}
$$

Note we take the expectation with respect to the dynamic model $q_{\theta}$ and action $a_{\phi}$. In the first loss we're just trying to maximize the value estimate by updating the actor. In the second loss we're updating the value function to be close to the value estimate.

### Implementation and results:

I've implemented a simple module for the RSSM approach [here](https://github.com/mauicv/world-model-rl). I managed to train the deep mind control task `dm_control/walker-walk-v0` in a a [google colab](https://colab.research.google.com/drive/1Lj1Bhg5vwQJAhS_Ehq5X_w6AF3MTxfnk) with only a T4 GPU and to a reward of around 600/700 in a day or so. I think it could be pushed further but I didn't try.

We can generate a imagined rollout by reconstructing the latent states using the decoder. Doing so gives us:

![imagined rollout](posts/rl-world-model/rssm-imagined-rollout.gif)

In turn we can also use the actor in the real world environment:

![real rollout](posts/rl-world-model/rssm-real-rollout.gif)


# Transformer state space model (TSSM):

The TSSM replaces the Recurrent neural network with a transformer. This section is largely based on [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/pdf/2202.09481) which introduces TSSM, but also on [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2303.07109) which similarly applies transformers to world modelling.

There are two main reasons we might want to use a transformer based world model.

- __1. Scale__: RNN's utilize a hidden state that has to be computed at each step in the rollout that we're training them on. Transformers on the other hand are designed to avoid this issue. Instead of having some hidden state thats passed back into the model on each step we instead pass all states into the model and allow future states to attend to past states. Because of this we don't need to perform the dynamic learning inside a for loop and instead everything is just pushed through the transformer in one go.
- __2. Long Term Dependencies__: The dynamic history in a RNN has to be encoded within the hidden state and this means the model has to know what to store in a past state that may become relevant in a future state. In contrast because the transformer uses an attention mechanism - future states can attend to previous states as they need. In this sense there is no hidden state bottleneck.

The first key difference to the RSSM world model is that transformers work best on discrete latent spaces. To this end we use a categorical encoding in the latent space instead of a continuous one. This just means instead of the encoder mapping to a vector of real values we instead map to $n$ one hot encoded vectors of length $m$. So each input image is represented by $n$ discrete random variables each with $m$ categories.

The second key difference is that when training the transformer you predict all the next states at once and you don't need to compute the hidden state at each stage. Suppose you have a rollout $\{(o_i,a_i,r_i)\}_{i\in\mathbb{N}}$ and you map the observations to latent states using the encoder to get $\{(s_i,a_i,r_i)\}_{i\in\mathbb{N}}$. Let's denote the transformer as $f$ then we can think of it acting on the entire sequence like so $f(\{(s_i,a_i,r_i)\}_{i})=\{(h_{i+1})\}_{i}$.

Under the hood each item in the sequence is embedded and then passed through a series of layers. At each layer the item can choose to attend (or look at) previous items in the sequence and update its own value dependent on them. The transformer outputs a sequence of hidden states $\{(h_{i+1})\}_{i}$ which we pass to prediction and reward models in the exact same way as the RSSM to obtain probability distributions of each. We then minimize the kullback leibler divergence between the predicted state distribution and the true state distribution.

Otherwise when generating imagined rollout we need to generate the sequence one by one same as the RSSM. Doing so also requires passing the entire sequence back in on every step unless you use a KV cache. Training the agent is the same and if you implement this process correct the agent training code should basically be agnostic of the world model code.

### Implementation and results:

The [world model rl](https://github.com/mauicv/world-model-rl) repo where I implemented RSSM also contains the implementation of TSSM. And [this](https://colab.research.google.com/drive/1VgJ7E-THAOO1kPk7UWgfpbI0kNF_6gTi#scrollTo=pEy1rLE6msSx) colab notebook contains the code to train an agent for the deep mind control task `dm_control/walker-walk-v0` on a T4 GPU.

_Note that I'm calling it TSSM but the TSSM authors never released the code so I'm not 100 percent sure how true it is to there implementation and I know of a couple of ways in which it differs from what they suggest in there paper._

An example of an imagined rollout:

![imagined rollout](posts/rl-world-model/tssm-imagined-rollout.gif)

Actor in the real world environment:

![real rollout](posts/rl-world-model/tssm-real-rollout.gif)


__Note__: Don't compare the RSSM and TSSM performance from the above rollouts as they're trained for different periods of time and so it's not a fair comparison. In general I found the RSSM to perform better than TSSM and there are a couple of reasons this might be the case.

- __1.__ The transformer world model reward target has many more gradients paths through the model than the recurrent model. This potentially leads to instability when training the agent. This is the argument put forward in [Do Transformer World Models Give Better Policy Gradients?](https://arxiv.org/abs/2402.05290) by Ma et al.
- __2.__ The transformer model is a bigger model than the RSSM and the size of the model comes with trade offs - in this case shorter rollouts. The RSSM is trained on longer (15 steps) rollouts than the transformer model is (10 steps).

__Relavant papers:__

- __1.__ [world models](https://arxiv.org/abs/1803.10122)
- __2.__ [world models blog post](https://worldmodels.github.io/)
- __3.__ [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
- __4.__ [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
- __4.__ [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2303.07109)
- __5.__ [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/abs/2202.09481)
