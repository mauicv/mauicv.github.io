__note__: *Relevant code for this post is [here](https://github.com/mauicv/openai-gym-solns)*

__note__: _2024 Alex: "I wrote this post a long time ago when I was starting learning about RL and so its pretty faulty due to me not knowing what I was talking about. For a slightly less lacking brain dump of content about rl please see [1](#/posts/rl-nes), [2](#/posts/rl-dqn), [3](#/posts/rl-pg), [4](#/posts/rl-ddpg), [5](#/posts/rl-wrld-model)_

---

## Policy Gradient Methods


The reinforcement learning problem is made up of a state space, and actor and a policy. The actor transitions between states in the state space on the basis of the actions it's taking at those states. The actions it chooses to take are selected by the policy. The policy is just some function that takes the state as input and spits out advice on the best action to take. The policy is usually modelled by a neural network which means we can lean on machine learning algorithms to improve it's performance over time. The class of methods I'm going to introduce here are Policy Gradient Methods. In particular I'm implementing REINFORCE.

__Note__: *Here I assume you know rudimentary details of how neural networks work and machine learning work. Like what they look like, that they're parameterized by lots of weights and biases. How you can compute the derivative w.r.t. these parameters using back-propagation. How the derivative of a loss function w.r.t. these parameters tells you what direction to change the parameters in order to improve the networks performance... These kinds of things.*


So one potential solution to the above problem is to take a policy function that takes as input the state the actor is in and then gives as output a distribution that tells us which actions are most likely to result in fulfilling the goal. Initially this function is just a guess, it just gives random probabilities for the best action. The goal of training is to encourage the policy function to get better and better at suggesting the best action to take at each state. So suppose your solving the [lunar lander](https://gym.openai.com/envs/LunarLander-v2/) environment and you record the actions that this function dictates at each state in the actors (the spaceships) trajectory. At the end you look through each of the actions and each of the states and ask which actions resulted in positive outcomes and which resulted in negative outcomes. You then want to encourage those that resulted in success and discourage those that didn't.

At each state we'll give the policy the location of the shuttle and it'll pass that data through each of it's layers and output a vector of values. We're going to assume the set of actions are discrete so, engine on or off, rather than continuous, engine fire at 60% or maybe 65% or 65.562%. This means that the vector of values above corresponds to probability of which engine to fire. So given a state we compute the action probabilities and then sample from these probabilities the action we take. Because of this during training we don't always do the same thing, we do some things much more often but occasionally we try stuff that the policy is mostly advising against. We then use back propagation to obtain the parameter change in the policy weights and biases that's going to result in the network suggesting actions that do well more than actions that do badly.

Denote a policy by $$\pi$$ and the set of parameters underlying it, a vector of weight and biases, $$\theta$$. The set of states and actions generated by taking a state, computing the policy, sampling a action, transition to a new state and repeating is an orbit, denote such an object $$\tau$$. These look like:

$$\tau_{\theta} = [(s_{0}, a_{0}), (s_{1}, a_{1}), ..., (s_{i}, a_{i}), (s_{i+1}, a_{i+1}), ....]$$

Where $$a_{i}$$ is an action sampled from the policy probability distribution $$\pi_{\theta}(s_{i})$$. If the system is deterministic then given a state and an action we can directly compute the subsequent state. So $$(s_{i}, a_{i}) \rightarrow s_{i+1}$$. If the system is not deterministic then you conduct some experiment and sample the next state from the probability distribution of states given by taking action $$a_{i}$$ in state $$s_{i}$$.

At the end of the orbit the actor has either achieved it's goal or failed to do so. If it's achieved it we have to go back through the set of actions it's taken and find a way of allocating rewards to each on the basis of how well it's performed.

When we decide we're going to encourage an action in a given state then we need first to know how the network changes with respect to it's parameters, the weights and biases. This change in parameter space is given by the derivative of the model output with respect to those parameters. Knowing this we can use an update rule that looks something like:

$$
\theta \rightarrow \theta + \sigma A\nabla \pi_{\theta}( a_{i}\| s_{i})
$$

Where $$\nabla \pi_{\theta}( a_{i}\| s_{i})$$ is the derivative of the policy w.r.t. $$\theta$$ __at the chosen action for that state__. Adding it to $$\theta$$ is like saying walk up or down the probability density hill so as to make that action more or less likely. A is the reward we decided to allocate that action and is positive if we think, $$a_{i}$$, resulted in a good outcome and negative if not. So then the above should make $$a_{i}$$ more likely if it was good and less if it was bad. Finally $$\sigma$$ is the learning rate.

#### The problem with the above

We don't always get the optimal solution. Suppose we have a system with only one state and two actions. One of those actions has a big reward and the other a little reward. Suppose the way the policy network is initialized means that it suggests the low reward over the high reward with a high probability. Ideally this shouldn't be a problem. Over the training the network should end up reassigning the probability towards the better reward action. Unfortunately because the policy is initially incorrectly biased towards the poor reward option we're going to get far more samples of this action than the other and because it has a positive, albeit lower reward the training will end up encouraging this action more. This is simply because it gets more samples for it. A sufficiently large amount of a small thing can be more than a small number of a big thing. It's a bit like failing to learn to do something a new and better way because you want to prioritize doing it a worse way that you know well. This means we have to counteract this behaviour by incorporating the policy probabilities themselves into the update rule.

So if an policy suggests an action and it returns a positive reward then we update in favour of that action dependent on how likely the policy was to suggest that reward. This way updates to low reward actions that the policy suggests a lot are balanced by the higher probabilities of the policy suggesting them. High reward actions that the policy isn't likely to suggest are boosted to make up for the smaller likelihood of sampling them. The way we do this is just to divide by the probability of selecting an action.

$$
\theta \rightarrow \theta + \sigma A\frac{\nabla \pi_{\theta}(a_{i}\|s_{i})}{ \pi_{\theta}(a_{i}\|s_{i})}
$$

There are a couple of ways to set A. It's the amount we're going to encourage the network to take that action next time it finds itself in the same state. So in other words is how you evaluate the quality of the action taken with respect to the outcome received. For example a naive approach would just have it be the a constant positive number if the task is completed correctly and a constant negative number if incorrectly.

The final issue we have is that the above function contains the derivative $$\nabla \pi_{\theta}(a_{i}\|s_{i})$$ which is inconvenient if we're using a machine learning frame work like [TensorFlow](https://www.tensorflow.org/) to implement this algorithm. This is because TensorFlow expects a loss functions that returns a scaler value. To solve this we can use the following:

$$
\nabla log(f) = \nabla {f}/f
$$

to get:

$$
\theta \rightarrow \theta + A\nabla log(\pi_{\theta}(a_{i}\|s_{i}))
$$

Which would make the loss function:

$$
log(\pi_{\theta}(a_{i}\|s_{i}))
$$

___

## The Algorithm


Assuming naive constant positive or negative rewards:

```
1. Sample an initial random state
2. Initialize and empty array to store episodic memory
3. For n steps:
  - Sample an action from the policy dependent on current state,
  - Take the action and move the actor into the new environment state
  - Record the action and new state in the episodic memory
4. If the actor was successful set A = 1 if unsuccessful set A = -1
5. Update the policy
6. Repeat for as many episodes as needed
```

Alternatives of the above discount the rewards back in time from the actor achieving reward by some value $$\gamma < 1$$. So if the actor is successful then the update on the action $$n$$ steps before the end of the episode is weighted by $$\gamma^n$$. This represents the fact that actions the actor takes just before it is successful should be rewarded more than actions taken further back in time.

You can also assign rewards not just for completing the task at hand but also at intermittent stages in the process. In this case you'd record those rewards in episodic memory at the same time as the state and action that led to them. You'd then discount that reward back in time from when it was obtained.