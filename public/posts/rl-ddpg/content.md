Up until now we’ve been using stochastic policies. DDPG introduces Deterministic Policy Gradients. The idea is a combination of DQN and Actor critic: We train a critic model: $Q_\theta(s, a)$ that learns the values of actions at each state, and train a policy, $\mu_{\theta}$ that tries to maximize the critic at each state. Because the policy is deterministic, we don’t have to try to maximize the expected reward and use importance sampling.

The optimization target is:

$$
J(\theta) = \int_S\rho^{\mu}(s)Q(s, \mu_\theta(s))ds
$$

where $\rho^\mu$ is the discounted state distribution given by: 

$$
\rho^\mu(s')=\int_{S}\sum_{t=1}^\infty \gamma^{t-1}p_0(s)p(s\rightarrow s',t, \mu)ds
$$

In the above, $p_0$  is the initial distribution of states, $p(s\rightarrow s', t, \mu)$ is the probability of transitioning from $s$ to $s'$.

The DPG theorem gives us:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{s\sim \rho^{\mu}}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s, a)|_{a=\mu_{\theta}(s)}]
$$

The proof ([here](https://proceedings.mlr.press/v32/silver14-supp.pdf)) is very like the proof for the policy gradient theorem.

The [**DDPG** paper](https://arxiv.org/abs/1509.02971) is really just the application of the above **DPG Theorem** to control problems using deep learning, hence the second D. Its essentially DQN plus an actor model that’s responsible for learning the action that maximizes the critic model for every state-action. The paper also:

- __1.__ Uses a __replay buffer__ in the same way as DQN.
- __2.__ Adds noise to the deterministic policy to encourage exploration.
- __3.__ Uses __soft updates__ for target copies of the learned models. This means we use a target actor and critic which is updated at a slower rate than the true critic and critic. Using these to compute the target value for the critic update at each step causes learning to be more stable.
- __4.__ __Batch normalization__ to manage inputs with different variances. (I actually couldn't get this method to work using batch norm and so removed it in my implementation.)

**Full algorithm:**

![ddpg algorithm](/posts/rl-ddpg/algorithm.png)

__Note:__ _I found DDPG very brittle and basically a nightmare to get working. It regularly seems to almost solve the environment and then suddenly collapse and seemingly forget all the progress it had made. The below works for the pendulum environment assuming you cut off training before collapse._

This gist is DDPG applied to [Pendulum-v1](https://gist.github.com/mauicv/d05aba08051c3b840ebbede160b28249) and this is DDPG applied to [Bipedal-walker-v2](https://gist.github.com/mauicv/0091534795880127103e9744b97f92d9).

# Twin Delayed Deep Deterministic policy gradients (TD3):

[**TD3**](https://arxiv.org/abs/1802.09477) aims to resolve some of the stability issues with **DDPG**:

- __1.__ **Twin Critics:** The critic in DDPG overestimates the value of actions. This is due to the policy being trained to maximize the critic causing a bias towards overestimation. To solve this Fujimoto et al, use 2 critics and when computing the target in the TD update we use the minimum action value predicted in order to try to underestimate the action value:
    
    $$
    y=r+\gamma \min_{i\in 1,2} Q_{w_i} (s', \mu_{\theta_i}(s'))
    $$
    
- __2.__ **Target Policy Smoothing:** Learning a deterministic policy by maximizing the critic is susceptible to inaccuracies induced by function approximation error in the critic. Essentially errors in the critic learning can create narrow peaks which the policy will overfit to. To solve this the authors add clipped noise to the actions when computing the TD target:
    
    $$
    y=r+\gamma Q_{\omega}(s', \mu_{\theta}(s')+\epsilon)
    \\
    \epsilon \sim clip(\mathcal{N}(0, \sigma), -c, +c)
    $$
    
- __3.__ **Delayed Updates of Target and Policy Networks:** I don’t fully understand this idea but according to the paper: *‘Value estimates diverge through overestimation when the policy is poor, and the policy will become poor if the value estimate itself is inaccurate.’* In general the idea is to update policy at a lower frequency than the critic, this gives the critic predictions a better chance to stabilise (reduce there variance) before the policy learns the valuable actions.

![td3 algorithm](/posts/rl-ddpg/td3-algorithm.png)

This gist is td3 applied to [Pendulum-v1](https://gist.github.com/mauicv/a6d6bc22c1b664e5496028159d40c95d) and this is it applied to [Bipedal-walker-v2](https://gist.github.com/mauicv/5a34dc0acb7620199aa7cd5e3011da0e).

Next: [World Models and RL](#/posts/rl-world-model)
