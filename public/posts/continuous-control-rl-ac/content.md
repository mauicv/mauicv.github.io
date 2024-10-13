___

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

We can do even better than the above by using the advantage $A^{\pi}(a_t, s_t) = Q^{\pi}(a_t, s_t) - V^{\pi}(s_t)$ instead of just $Q^{\pi}(a_t, s_t)$ when updating the actor (policy) - step 4. Intuitively the advantage is the expected value of choosing the action $a_t$ compared to the expected value of that state.

The policy update rule becomes:

$$
\theta\leftarrow\theta+\alpha A(a_i,s_i)\nabla_\theta\log\pi(a_i|s_i)
$$

Note that $Q^{\pi}(a_t, s_t)=r_t+V^{\pi}(s_{t+1})$ and so hence: $A^{\pi}(a_t, s_t) = r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$. Hence instead of learning $Q$ and $V$ we can just learn $V$ and use:

$$
\theta\leftarrow\theta+\alpha \nabla_\theta\log\pi(a_i|s_i)[r_t + V(s_{t+1})-V(s_t)]

$$

Note that with the above the new critic update rule is:

$$
V(s_t) \leftarrow V(s_t)  + \alpha (r_{t+1} + \gamma V(s_{t+1}) - V(s_t))
$$

[This github gist](https://gist.github.com/mauicv/c8650ddc6aaf9e1deb9f33dc2f14ccc3) contains an example of advantage actor critic applied to CartPole.

Next: [Deep Deterministic Policy Gradients](#/posts/continuous-control-rl-ddpg)