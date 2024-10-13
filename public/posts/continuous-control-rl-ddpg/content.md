___

# Deterministic Policy Gradients

Up until now we’ve been using stochastic policies. DPG introduces Deterministic Policy Gradients. The idea is a combination of DQN and Actor critic: We train a critic model: $Q_\theta(s, a)$ that learns the values of actions at each state, and train a policy, $\mu_{\theta}$ that tries to maximise the critic at each state. Because the policy is deterministic, we don’t have to try to maximise the expected reward and use importance sampling.

The optimisation target is:

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

DPG can be performed on and off policy. Even better if we use off-policy, because of the deterministic nature of the policy we don’t have to worry about importance sampling.

# Deep Deterministic Policy Gradients

The **DDPG** paper is really just the application of **DPG** to control problems using deep learning, hence the second D. Its essentially DQN plus an actor model that’s responsible for learning the action that maximises the critic model for every state-action. The paper also use a set of additional improvements:

- __1.__ Because its off-policy we can use the replay buffer in the same way as DQN.
- __2.__ Same actor-critic pattern as in the Actor Critic algorithm.
- __3.__ Add noise to the deterministic policy to encourage exploration.
- __4.__ Soft updates using target copies of the learned models.
- __5.__ Batch normalisation to manage inputs with different variances

**Full algorithm:**

![ddpg algorithm](/posts/continuous-control-rl-ddpg/algorithm.png)

DDPG is very brittle and basically a nightmare to get working. It regularly seems to almost solve the environment and then suddenly collapse and seemingly forget all the progress it had made. The below works for the pendulum environment assuming you cut off training before collapse.

This gist is ddpg applied to [Pendulum-v1](https://gist.github.com/mauicv/d05aba08051c3b840ebbede160b28249) and this is it applied to [Bipedal-walker-v2](https://gist.github.com/mauicv/0091534795880127103e9744b97f92d9).

# Twin Delayed Deep Deterministic policy gradients (TD3):

**TD3** aims to resolve some of the issues with **DDPG**:

- __1.__ **Twin Critics:** The critic in DDPG overestimates the value of actions. This is due to the policy being trained to maximise the critic causing a bias towards overestimation. To solve this we use 2 critics and when computing the target in the TD update we use the minimum action value predicted in order to try to underestimate the action value:
    
    $$
    y=r+\gamma \min_{i\in 1,2} Q_{w_i} (s', \mu_{\theta_i}(s'))
    $$
    
- __2.__ **Target Policy Smoothing:** Policies can overfit to narrow peaks of the critic function. To solve this we add clipped noise to the actions when computing the TD target:
    
    $$
    y=r+\gamma Q_{\omega}(s', \mu_{\theta}(s')+\epsilon)
    \\
    \epsilon \sim clip(\mathcal{N}(0, \sigma), -c, +c)
    $$
    
- __3.__ **Delayed Updates of Target and Policy Networks:** I don’t fully understand this idea but according to the paper: *‘Value estimates diverge through overestimation when the policy is poor, and the policy will become poor if the value estimate itself is inaccurate.’* In general the idea is to update policy at a lower frequency than the critic, this gives the critic predictions a better chance to stabilise (reduce there variance) before the policy learns the valuable actions.

![td3 algorithm](/posts/continuous-control-rl-ddpg/td3-algorithm.png)

This gist is td3 applied to [Pendulum-v1](https://gist.github.com/mauicv/a6d6bc22c1b664e5496028159d40c95d) and this is it applied to [Bipedal-walker-v2](https://gist.github.com/mauicv/5a34dc0acb7620199aa7cd5e3011da0e).

Next: [World Models and RL](#/posts/continuous-control-world-model-rl)
