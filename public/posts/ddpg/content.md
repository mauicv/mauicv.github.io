__note__: *Relevant code for this post is [here](https://github.com/mauicv/BipedalWalker-v2-ddpg)*

---


## Continuous Control

It's been a bit of time since I last posted. I've had a lot of work on recently but I've started working on a new project that's been the motivation for learning a lot of this reinforcement learning material. Namely I want to build robots and train the control systems for these robots using reinforcement learning algorithms. Thus far the algorithms I've discussed here have been for discrete action spaces. This doesn't work for continuous control systems of the type I'm hoping to create. This post will be a rundown of deep deterministic policy gradients (DDPG) which is a reinforcement learning algorithm suited for continuous control tasks. This means stuff like motors that can apply a torque with a continuous range of values rather than one that is either on or off. In this post I'm going to talk about theory but mostly only to explain the stuff that it took me a while to understand.


___

## Why DDPG?

So I think it is actually possible to use [REINFORCE]({% post_url 2020-07-22-policy-gradient-methods %}) to do continuous control. What you'd do is have a policy (function that given a state suggests an action to take) that generates a continuous probability distribution. You could do this by having the model output a mean and a variance and then sample an action from the distribution those describe. You'd have to use importance sampling to ensure the actions the policy is more likely to suggest aren't overweighted in the reinforcement learning updates. Then at the end instead of using that probability distribution to sample actions you'd just take the mean itself as the action and ignore the variance. This is very similar to the discrete action case but instead of a probability vector your having to get the model to describe a continuous distribution instead.

I tried this and struggled with it. I'm not sure why exactly but in the end I read about DDPG and the consensus seemed to be that it was best for continuous control tasks. DDPG differs from REINFORCE in a number of ways. It's similar however in that it still uses a policy. The policy is the core object we're interested in here. Initially it's just going to suggest random actions but by the end of the training it should have learned to solve the environment in order to maximise rewards for the agent. The differences are as follows.

- Firstly DDPG requires using a critic. A critic is a function that takes a state and action and then returns the expected discounted reward the agent will receive over the whole orbit if they take that action and then from then on out take the action dictated by the policy. So the critic at any state is a function of just the possible actions at that state. Everything else is fixed including all the future actions as given by the policy. What this means is that the critic should give an estimate of the outcome for each possible action given a state. To find the best action you'd just choose the action that maximizes the critic. The critic aims to estimate $$C(s_i, a_i) = \mathbb{E_{p}}(\sum(\gamma^i *r_i)| s_i, a_i)$$.
$$\gamma$$ here is the discount factor. It weights rewards in the immediate future more heavily that rewards in the distant future. The idea being that a reward received soon after an action is more likely to be as a result of that action than a rewards received long after. The important thing to note here is that if you know the the values $$(s_i,a_i)$$, $$(s_{i+1},a_{i+1})$$ and $$r_i$$ which are a pair of consecutive state action pairs in an orbit and the true reward for transitioning from state $$s_i$$ to $$s_{i+1}$$ then the critics estimate should satisfy:

$$C(s_i, a_i) = r_i + \gamma*C(s_{i+1}, a_{i+1})$$

- Secondly instead of recording a set of memories for just one episode and then updating your policy on the basis of just those you instead have a memory buffer which stores the relevant states, actions and rewards over many episodes. This ends up being a [circular buffer](https://en.wikipedia.org/wiki/Circular_buffer) where the entries look like: `(state, next_state, action, reward)`. When ever you want to train the policy and critic your going to take a random batch of samples from this buffer as the training data.

- Finally DDPG uses target models which are basically copies of the actor (policy) and critic that are updated slower than the actual actor and critic. I'm going to ignore these for now and mention them at the end but basically they deal with the fact that the actual actor and critic can change quite a lot on each training step which leads to instability in the learning.


When training DDPG We use the policy model, plus some noise for exploration, to sample from the environment the values `(state, next_state, action, reward)` that we then place in the replay buffer. You might ask why we need the policy model seeing as we can select actions to take given any state by finding the action that maximises the critic. The reason this typically isn't feasible is due to the size of the action space. If you have a discrete action space there are usually a smaller set of possible actions you can take, think 2 engines for the Lunar-lander each is on or off, making 4 possible combinations of values in the action space to maximize the critic over. In this case it's feasible to maximise over. Whereas in a 2 dimensional continuous action space to maximise correctly you'd have to subdivide each dimension into small bits and then consider all the combinations which would end up being $$n^2$$ where $$n$$ is the number of subdivisions you make. If now you consider $$m$$ dimensions it's then $$n^m$$. This quickly becomes prohibitive.

So maximising over the critic takes too long and instead what you do is use this policy model that tries to maximise the critic as it changes over the training period by climbing it using gradient ascent. This basically means that instead of starting from scratch every time you search for that maximising value you instead use the policy to track the correct value over time. The issue with this approach is that the policy may get stuck in a local maximum.

___

## Training:

Training with DDPG uses two rounds, the first updates the critic using temporal differences and the second updates the policy by gradient ascent. Bear in mind that at each training step the set of values we have is sampled from the replay buffer and takes the form: `(state, next_state, action, reward)`

### Critic:

To train the critic we just take the initial prediction of the critic for a selected `(state, action)` pair. Call this value $$c(s_{i}, a_{i})$$. We then get the `next_action` value by plugging in the `next_state` into the policy and then computing the next critic value given as $$c(s_{i+1},a_{i+1})$$ where $$a_{i+1} = P(s_{i+1})$$ and $$s_{i+1}$$ is the `next_state`. Now we have the critic value for `(state, action)` and the critic value for `(next_state, next_action)`. If $$c(s_{i}, a_{i})$$ is accurate then it should equal $$\gamma * c(s_{i+1},a_{i+1})$$ plus the reward obtained for that state. The full equation:

$$c(s_{i}, a_{i}) = r_{i} + \gamma * c(s_{i+1},a_{i+1})$$

If the critic is untrained then it won't equal the above but we now have a target to train the critic towards. Namely the difference in the above:

$$t_i = r_{i} + \gamma * c(s_{i+1},a_{i+1}) - c(s_{i}, a_{i})$$

To update the critic on each training run your going to minimize the above using gradient descent over the batch of samples you've taken from the replay buffer.

### Actor:

To train the actor/policy is simply just a case of asking it to climb the critic function. So if you take the gradient of $$C(s_i, P_{\omega}(s_i))$$ with respect to the policy parameters given by $$\omega$$ you can compute how best to change $$P$$ in order to increase the value of $$C$$. Again you do this over the batch of samples take from the replay buffer.

### Target Actor and critic

So the above is a slight simplification in that it doesn't talk about the target models we use to add stability. Basically when you compute  

$$t_i = r_{i} + \gamma * c(s_{i+1},a_{i+1}) - c(s_{i}, a_{i})$$

instead of computing $$c(s_{i+1},a_{i+1}) = c(s_{i+1},p(s_{i+1}))$$ we use:

$$c_{targ}(s_{i+1},a_{i+1}) = c_{targ}(s_{i+1},p_{targ}(s_{i+1}))$$

where $$c_{targ}$$ and $$p_{targ}$$ are the target critic and target actor and are just copies of the critic and actor that are updated much slower. By this I mean whenever you update the actor and critic, you then update the target actor and target critic like so:

$$\omega_{a_{targ}} \leftarrow \omega_{a}\tau +  \omega_{a_{targ}}(1 - \tau)$$

$$\omega_{c_{targ}} \leftarrow \omega_{c}\tau +  \omega_{c_{targ}}(1 - \tau)$$

where $$\omega_{c_{targ}}$$ and $$\omega_{a_{targ}}$$ denote the model parameters for the target actor and target critic and $$\tau$$ is some small value usually $$0.05$$

___

### Full algorithm:

The full algorithm taken from the [original paper](https://arxiv.org/pdf/1509.02971.pdf) by Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,Tom Erez, Yuval Tassa, David Silver & Daan Wierstra and researched at google Deepmind is as follows:

![ddpg-algo](/posts/ddpg/ddpg-algo.png)

___

## Outcomes

It's kind of hard to believe the above works. So here is proof it does:

![ddpg-bipedal-walker](/posts/ddpg/ending.gif)
