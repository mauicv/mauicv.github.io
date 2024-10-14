[Natural Evolutionary Strategies](#/posts/rl-nes) are nice because you only care about the entire reward for a whole environment rollout and not individual actions taken within a rollout. This makes it easy to implement. However what if we can get more signal by looking at the outcomes of individual actions taken given specific states in a rollout. 

Ultimately we want a way of taking a state and turning it into an action. We want this action to maximize the cumulative rollout reward. Ideally we'd just have $X$ and $Y$ where $X$ is a dataset of states and $Y$ is the actions to take in those states. Then we could just fit a model that generates those actions in those state. However in the RL paradigm we don't have such a training dataset we just have the environment. Our dataset is made up of sequences of state, action and reward values sampled from the environment.

If we can train a model $Q$ that takes actions and states and output an accurate estimate of the value (expected cumulative reward over future rollout) of the action then we actually have a way of solving the environment. We do this by simply selecting the action that maximizes $Q$'s output at each state. This is Q-Learning.

__Note__: _The above works well for small sets of discrete actions however if we're applying this to continuous action problems there is an obvious issue. Namely, computing the action that maximizes $Q(s, a)$ becomes computationally expensive._

## Reward discounting and The reward Function

Given a rollout $(s_t, a_t, r_t)_{t=0}^{N}$ We define the future discounted reward of a state $s_i$ in this rollout as:

$$
R(s_i)=\sum_j\gamma^j r_{j+i+1}
$$

Where $0<\gamma<1$ and is the discount factor. This is used to weight rewards based on how close they where to the action that carried them out. So if an action is taken and a lot of reward is obtained soon after that reward is more significant that if a large amount of reward occurs much further down the line.

Now we’ve defined the discounted rewards we can define the target we're trying to optimize. We want to maximise the following reward function:

$$
J(\theta) = \sum_{s\in S}d^\pi(s)V^\pi(s)=\sum_{s\in S}d^\pi(s)\sum_{a\in A}Q^\pi(s, a)
$$

Where

- __A.__ $d^\pi(s)$ is the stationary distribution of $\pi$ in the environment. The stationary distribution refers to the distribution of states if you let the policy run for infinite time. In the case of finite MDP we can extend them to infinite processes by looping them on episode end, allowing us to use the same analysis.
- __B.__ $V^{\pi}(s)$ is the expected return of a policy $\pi$ from state $s$. So $\mathbb{E}_{a\sim \pi}(R(S)|S=s)$
- __C.__ And $Q^{\pi}(s,a)$ is the same as $V^{\pi}(s)$ but taking actions into account as well. So $Q^\pi(s,a) = \mathbb{E}_{a\sim \pi}(R(S)|S=s, A=a)$

__Note__: _This is a similar idea to the fitness function we used in NES_

## The Bellman Equation:

The idea of Q-learning is that we just learn $Q(s, a)$ directly. In particular we can then just define the policy as $\pi(s) = \argmax(Q(s,\cdot))$. The $Q$ and $V$ functions follow a nice recursive relationship, namely the bellman equation:

$$
Q(s_t,a_t)=R_{t+1} + \gamma V(s_{t+1}) = R_{t+1} + \gamma \max_{a\in A} Q(s_{t+1},a_{t+1}) 
$$

If $Q(s_t,a_t)$ are not optimal then we can extract an update rule from the above by taking the difference of each term in the equality. We introduce a hyper parameter $\alpha$ that just controls the size of the update. 

$$
\begin{equation}
Q(s_t,a_t) \leftarrow Q(s_t,a_t)  + \alpha (R_{t+1} + \gamma \max_{a\in A} Q(s_{t+1},a_{t+1}) - Q(s_t,a_t))
\end{equation}
$$

## The Algorithm:

Thus the Q-learning algorithm is as follows:

- __1.__ Initialise $t_0$ and $s$. 
- __2.__ Compute $a_0=\argmax(Q(\cdot, s_{0}))$
- __3.__ Sample next state , $s_1$, using $a_0$ and get reward $r_1$
- __4.__ Update the $Q$ function using equation (1)
- __5.__ Repeat

This algorithm can be implemented without any deep learning elements and neural nets. All you require is a lookup table that maps states and actions to there $Q$ values. This works well but gets harder to do once you reach sufficiently large state-action spaces. Hence the next idea: DQN

# DQN (Deep Q Network)

DQN, introduced in [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), takes Q-learning and adds a deep neural network. The key reason for doing so is because NNs are more suited computationally to functional approximation over large state-action spaces than something like a dictionary look up table, namely due to the fact that they interpolate between inputs.

In order to do this we approximate the the quality function with a neural network, $Q_\theta(s, a)$. The loss function becomes:

$$
L(\theta)=\mathbb{E}_{a_i,s_i,a_{i+1},s_{i+1},r_{i+1}\sim P}[(r_{t+1}  + \gamma \max_{a'\in A} Q_{\theta'}(s_{t+1},a') - Q_\theta(s_t,a_t))]
$$

Where $\theta'$ are frozen parameters. I.e. (see comment later!)

The above will be useful later because its the exact same loss we see in the critic function used in **actor-critic** and **DDPG.**

In order to improve stability the authors of the Deep Q-learning paper also 

- __1.__ Use an experience replay buffer which stores a history of policy rollouts. i.e. tuples of $\{(s_i, a_i, r_{i+1}, a_{i+1}, s_{i+1})\}_{i\in\mathbb{N}}$.
- __2.__ Only update the $Q$ model intermittently. So we freeze $Q$ and accumulate the changes in this and then transfer it back to $Q$ after a certain number of steps. We always choose the action with respect to the frozen $Q$ model. Doing so reduces oscillations.

[This](https://gist.github.com/mauicv/2f2b3afea4de11fee343e4863cf354c3) colab notebook applies DQN to the CartPole environment. It should be said that it's not always successful - but later methods significantly improve stability.

Next: [Policy Gradients methods](#/posts/rl-pg)