
Q-learning approaches learn the quality of each action in each state in order to compute the optimal action from that state, thus they indirectly learn the policy. Policy gradient methods however learn the policy directly.

Recall we're trying to maximize the following objective:

$$
J(\theta) = \sum_{s\in S}d^\pi(s)V^\pi(s)=\sum_{s\in S}d^\pi(s)\sum_{a\in A}Q^\pi(s, a)
$$

Where

- __1.__ $d^\pi(s)$ is the stationary distribution of $\pi$ in the environment. The stationary distribution refers to the distribution of states if you let the policy run for infinite time. In the case of finite MDP we can extend them to infinite processes by looping them on episode end, allowing us to use the same analysis.
- __2.__ $V^{\pi}(s)$ is the expected return of a policy $\pi$ from state $s$. So $\mathbb{E}_{a\sim \pi}(R(S)|S=s)$
- __3.__ And $Q^{\pi}(s,a)$ is the same as $V^{\pi}(s)$ but taking the action into account as well. So $Q^\pi(s,a) = \mathbb{E}_{a\sim \pi}(R(S)|S=s, A=a)$

Its difficult to compute the derivative of the above due to the $d^\pi$ term and its dependency on $\pi$. However we can use the policy gradient theorem (see [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#policy-gradient-theorem) for derivation).

$$
\nabla_\theta J(\theta)=\nabla_\theta\sum_{s\in S}d^\pi(s)\sum_{a\in A}Q^\pi(s, a)\pi(a|s) \propto \sum_{s\in S}d^\pi(s)\sum_{a\in A}Q^\pi(s, a)\nabla_\theta\pi(a|s)
$$

Multiplying the above by $\pi(a|s)/\pi(a|s)$ we get:

$$
\sum_{s\in S}d^\pi(s)\sum_{a\in A}\pi(a|s)Q^\pi(s, a)\frac{\nabla_\theta\pi(a|s)}{\pi(a|s)}
$$

And then due to $ln(g(x))'=g'(x)/g(x)$:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}_{s\sim d^{\pi}, a\sim\pi}[Q^\pi(s, a)\nabla_\theta\log{\pi(a|s)}]
$$

Where the expectation follows as the stationary state is a distribution along with $\pi$. The above gives us something that we can estimate by drawing samples from environment rollouts.

## The REINFORCE algorithm:

From the policy gradient theorem we have:

$$
\nabla_\theta J(\theta) \propto \mathbb{E}_{s\sim d^{\pi}, a\sim\pi}[Q^\pi(s, a)\nabla_\theta\log{\pi(a|s)}]
$$

If we take a rollout of an entire environment trajectory for the policy $\pi$ we have set of states, actions and rewards: $\{(s_i, a_i,r_i)\}_{i\in\mathbb{N}}$. In particular for a specific $s_i$ we can estimate the above using: $Q^\pi(s_i,a_i)=\sum_j \gamma^{j} r_{j-i+1}=R(s_i)$ and also $\nabla_\theta\log\pi(a_i|s_i)$. Hence if we perform a roll out of the environment we can compute for each step an update that will minimise the loss, and this is the idea of REINFORCE:

**Algorithm**

- __1.__ Perform rollout and collect a sequence of state, action, reward tuples
- __2.__ Compute $R(s_i)$ for each instance in the sequence
- __3.__ For each item in the sequence update: $\theta\leftarrow\theta+\alpha R(s_i)\nabla_\theta\log\pi(a_i|s_i)$ where $\alpha$  is a learning rate.

### Intuitive Explanation:

One way of thinking about the above REINFORCE is as follows. 

- __1.__ In general we want to make actions that increase the discounted reward more likely and ones that don’t less likely. 
- __2.__ One way of doing this is to use the following update rule: $\theta \leftarrow\theta+R(s)\nabla_\theta\pi_\theta(a|s)$. The intuition behind this is simple: $\nabla_\theta\pi_\theta(a|s)$ is the direction to change $\theta$ in order to make $a$ more likely. Thus if $R(s) > 0$ the update will increase the probability of $a$ and if its less than $0$ it’ll decrease the probability.
- __3.__ There is a problem with the above however. If we sample from the orbits according to $\pi_{\theta}$ then we’ll bias the data by the policy. In other words the policy randomly at initialisation preference some action. This action might itself have positive reward. Because, when we perform rollouts we’re going to sample that action more often, it’ll become over represented in the training data and we might miss out other actions that are better because of this.
- __4.__ The answer to the above is to weight the significance of action updates dependent on how likely they are. To do this we normalise by there probability:
    
    $$
    \theta\leftarrow\theta+R(s)\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}
    $$
    
    which using the rule for log differentiation we get: $\theta\leftarrow\theta+R(s)\nabla_\theta\log\pi_\theta(a|s)$. Which is the same as the rule derived above.


This [gist](https://gist.github.com/mauicv/43ba180044b49a065ec30390e5189b3d) is an example of REINFORCE applied to the CartPole environment. It's still a little unstable but much better then q-learning.

# Actor Critic methods

REINFORCE and surprisingly well however its pretty unstable due to the variance of the rollout samples. The instability stems from the the sample estimate of $R(s_0)$ which can vary wildly between rollouts. One option is to get a more stable estimate of $R(s)$ by learning it in the same way we learn the $Q(a|s)$ function in Q-learning, i.e. using temporal difference methods and the bellman equations.

The on-policy algorithm works as follows:

- __1.__ Initialise $t_0$ and $s$. 
- __2.__ Compute $a_0=\argmax(Q(\cdot, s_{0}))$
- __3.__ Sample next state, $s_1$, using $a_0$ and get reward $r_1$
- __4.__ Update the Policy or Actor, $\pi$, using REINFORCE update rule:

$$
\theta\leftarrow\theta+\alpha Q(a_i, s_i)\nabla_\theta\log\pi(a_i|s_i)
$$

- __5.__ Update the $Q$ function (the critic) using TD learning:
    
    $$
    Q(s_t,a_t) \leftarrow Q(s_t,a_t)  + \alpha (R_{t+1} + \gamma \max_{a\in A} Q(s_{t+1},a_{t+1}) - Q(s_t,a_t))
    $$
    
- __6.__ Repeat

# Advantage Actor Critic

We can do even better than the above by using the advantage $A^{\pi}(a_t, s_t) = Q^{\pi}(a_t, s_t) - V^{\pi}(s_t)$ instead of just $Q^{\pi}(a_t, s_t)$ when updating the actor (policy, step 4 in actor-critic). Intuitively the advantage is the difference in expected value of choosing the action $a_t$ compared to the expected value of that state on its own.

Note that $Q^{\pi}(a_t, s_t)=r_t+V^{\pi}(s_{t+1})$ and so hence: $A^{\pi}(a_t, s_t) = r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$. Hence instead of learning $Q$ and $V$ we can just learn $V$ and use:

$$
\theta\leftarrow\theta+\alpha \nabla_\theta\log\pi(a_i|s_i)[r_t + V(s_{t+1})-V(s_t)]

$$

Note that with the above the new critic update rule is:

$$
V(s_t) \leftarrow V(s_t)  + \alpha (r_{t+1} + \gamma V(s_{t+1}) - V(s_t))
$$

We use the advantage in the policy update like so:

$$
\theta\leftarrow\theta+\alpha A(a_i,s_i)\nabla_\theta\log\pi(a_i|s_i)
$$

[This github gist](https://gist.github.com/mauicv/c8650ddc6aaf9e1deb9f33dc2f14ccc3) contains an example of advantage actor critic applied to CartPole.

Next: [Deep Deterministic Policy Gradients](#/posts/rl-ddpg)