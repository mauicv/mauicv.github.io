___

# RL for continuous control Introduction

This is a first post, covering natural evolutionary strategies, in a set of posts detailing RL methods for continuous control. Its mostly a content dump of notes I've written while learning the area but perhaps someone will find it helpful.

## What is RL?

In RL we have an environment $P$ and an actor, $A$. The environment is typically made up of states and some kind of transition rule which maps between states conditional on an action. So if the environment is in state $s_1$ and we take action $a_1$ in that state we use the transition rule $s_2\leftarrow P(s_1, a_1)$ to obtain the new state. Note that this transition rule might be deterministic or stochastic. Typically the environment will also output a reward after each transition as well as any terminal conditions that may occur.

The actor is a function that maps states to actions. We then use the transition rule to sample from the environment a new state $s$, a reward $r$, and any terminal conditions that might exist and be triggered, $t$. The actor is either deterministic $a = \pi_\theta(s)$ or stochastic $a\sim\pi_\theta(a|s)$. 

Typically the environment is the input of an RL algorithm and the actor or policy is the output, or the thing we want to learn. Usually it is a neural network or model of some kind. (In our case always a neural network). Some algorithms include other parts to this, such as critics which learn the relative value of actions given states or even world models which learn models of the environment itself.

A rollout of the environment for a given policy refers to the sequences $\{(s_i,a_i,r_i)\}_{i\in\mathbb{N}}$ where $a_i=\pi(s_{i-1})$ and $s_i$ and $r_i$ where computed from $P(s_{i-1}, a_i)$ or sampled from $s_i,r_i\sim P(s, r|s_{i-1}, a_{i})$. Note that the environment is usually given as a Markov decision process with transition probabilities given by $s_i,r_i\sim P(s, r|s_{i-1}, a_{i})$.

## What is Continuous Control?

Continuous control refers to RL problems for which the agent outputs continuous responses to the environment state rather than discrete ones. I'm abusing the notion slightly because I'm really interested in robotic control problems - and in particular I'm assuming continuous control with dense reward signals. This means that the agent is continuously moving in the space and every movement get some kind of positive or negative reward.

## What is Model Free RL?

Model free RL basically involves any RL algorithm that doesn’t explicitly try to model some aspect of the environment directly. Instead the actor learns from direct interaction with the environment. We'll start this series by focusing on model-free approaches but the last posts will deal with methods that do use world models to solve the environment. Note that model free is nice because you don't need to train a model of the environnement which is hard and expensive. However:

- __1.__ Model free is not sample efficient - It requires lots of sampling steps which is fine if your training in simulations on a computer but not if you're doing it in reality.
- __2.__ Using a simulation to train a model-free method means you have to write a simulation for the real world environment your trying to solve and also means accounting for any gap between the real world and the simulation.

## Natural Evolutionary Strategies (NES):

I think the best place to start with RL is [NES](https://arxiv.org/abs/1106.4487) because its quite an intuitive approach and if your trying to implement this stuff it ends up being pretty easy to get working.

The Core idea here is to model an evolutionary process in which you generate a population of models from a parent and then update the underlying parent model on the basis of which permutations perform best. In reality you basically perform a stochastic sampling of the parameters numerical gradients over the cumulative reward for episodes.

## The fitness of a solution

To start with define the fitness random variable $F(\theta)$ as the result of running a rollout of the environment with a policy parameterised by $\theta$ and accumulating the reward over it. The sum of rewards in this way is the fitness of specific $\theta$ for solving the problem.

This is what we want to perform gradient ascent on. We start with a parent solution given by $\theta$ and sample from a population of its children. Later we’ll define the population to be normally distributed but one could choose other distributions.

So the object of interest is $\nabla_\theta \mathbb{E}_{\hat{\theta}\sim p_{\theta}}[F(\hat{\theta})]$ where $p_{\theta}$ is a distribution of solutions that we are trying to improve.

We can use the log-likelihood trick to get:

$$
\begin{align}
\nabla_\theta \mathbb{E}_{\hat{\theta}\sim p_{\theta}}[F(\hat{\theta})] &= \nabla_\theta \int_{\theta} p_{\theta}(\hat{\theta})F(\hat{\theta})d\theta \\

&= \int_{\theta} \frac{p_{\theta}}{p_{\theta}} \nabla_\theta p_{\theta}(\hat{\theta})F(\hat{\theta})d\theta \\

&= \int_{\theta} p_{\theta}\nabla_\theta \log p_{\theta}(\hat{\theta})F(\hat{\theta})d\theta \\

&= \mathbb{E}_{\hat{\theta}\sim p_{\theta}}[\nabla_\theta \log p_{\theta}(\hat{\theta})F(\hat{\theta})]
\end{align}
$$

**Note:** *A clearer way of writing the derivative would be $\nabla_\theta p(\hat{\theta}, \theta)$ where $\hat{\theta}$ is sampled from $p_\theta$. This is a reparameterization trick similar to what’s done in VAEs. Importantly we don’t need to worry about the derivative of $F(\hat{\theta})$ because it’s not directly dependent on $\theta$.*

If we assume that $p_{\theta}$ is a normal distribution, then we can parameterize it by its mean, and optionally its variance. The definition of normal distribution is:

$$
\begin{align}

p_{\theta}(\hat\theta)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{\hat\theta - \theta}{\sigma})^2} \\
\end{align}
$$

Hence we have:

$$
\begin{align}

\log p_{\theta}(\hat\theta)&= \log {\frac{1}{\sigma\sqrt{2\pi}}} - \frac{1}{2}(\frac{\hat\theta - \theta}{\sigma})^2

\\

\nabla_\theta \log p_{\theta}(\hat\theta)&=-\nabla_\theta\frac{1}{2}(\frac{\hat\theta - \theta}{\sigma})^2 = \frac{\hat\theta - \theta}{\sigma^2}

\end{align}
$$

(6) follows because $\log(a\cdot b)=\log(a)+\log(b)$ and $\log(e^x)=x$.

Because  $\hat\theta$ is normally distributed around $\theta$ we can write $\hat\theta = \theta+\sigma\epsilon$ where $\epsilon\sim\mathcal{N}(0, 1)$.  Hence substituting into the above we get $\frac{\epsilon}{\sigma}$ and substituting this into (4) we get:

$$
\mathbb{E}_{\hat{\theta}\sim p_{\theta}}[\nabla_\theta \log p_{\theta}(\hat{\theta})F(\hat{\theta})] = \frac{1}{\sigma}\mathbb{E}_{\epsilon\sim p_{\epsilon}}[\epsilon F(\theta + \sigma\epsilon)]
$$

*An added bonus here is that to compute the gradient you only need the value $F(\theta+\sigma\epsilon)$. If we set the same random seed across all processes then $\epsilon$ can be computed both in the main thread that computes the gradient after collecting all the $F(\theta+\sigma\epsilon)$ values and also in the sub processes that compute the individual $F(\theta+\sigma\epsilon)$. This means we don’t have to communicate large gradient vectors between processes.*

### Implementation:

The following github gists showcase implementations that solve the CartPole environment and the Ant-v4 environment respectively

- __1.__ [CartPole Environment (Discrete control)](https://gist.github.com/mauicv/8b6c51f222a35d9abdebe82487360966)
- __2.__ [Ant-v4 Environment (Continuous control)](https://gist.github.com/mauicv/3b23fe1ed823d3a16d12ce361fa8482b)

Because of its stability and ease of implementation I think NES is a great place to start when trying to solve a continuous control RL environment. I actually think in many cases it’s the fastest solution. Not because the runtime is smaller but because theres almost no hyper-parameter tuning required which reduces the overhead from running multiple experiments. If you care about delivery (and not pulling your hair out) rather than SOTA this might be your best course of action! I remember when I first got into RL spending ages trying to get various policy gradient methods to work for continuous control and getting more and more frustrated with intermittent results. Eventually I read the NES paper and implemented it and worked on the first try. Here is the results of this [method applied to Ant-v4](https://github.com/mauicv/evo-ant).

![evo-ant-v4](/posts/continuous-control-rl-nes/evo-ant.gif)

Next: [Q-learning](#/posts/continuous-control-rl-q-learning)

