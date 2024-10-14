
__note__: *Relevant code for this post is [here](https://github.com/mauicv/BipedalWalker-v2-ddpg)*

__note__: _2024 Alex: "I wrote this post a long time ago when I was starting learning about RL and so its potentially pretty faulty due to me not knowing what I was talking about. For a slightly less lacking brain dump of content about rl please see [1](#/posts/rl-nes), [2](#/posts/rl-dqn), [3](#/posts/rl-pg), [4](#/posts/rl-ddpg), [5](#/posts/rl-wrld-model)_

---


In a previous [post](/posts/ddpg/) I gave a rough explanation of DDPG theory. Here for prosperity, I list the stupid mistakes that confounded me for far longer than they should while implementing the DDPG algorithm:

### Accidentally bounding the critic 🤦‍♂️

When creating the critic network I copied and pasted the actor network. In doing so I accidentally forgot to remove the `tanh` activation that the final layer of the actor uses. This means the critic could at most predict a total of reward between `-1` to `1` for the entire episode given any state and action pair! The reward for the bipedal walker environment much greater than 1 so if the critic is only ever returning at most 1 then it's ability to guide the actor is severely stunted.

### Mismatched actor and target actor 🤦

In order to debug the issues with DDPG applied to the bipedal walker environment I implemented the cart pole environment as well as it's a simpler environment in which it's easier to spot errors. For whatever reason the range of values the actions can take are different between each environment. When I went back to the bipedal environment I kept the high action bound from the pendulum environment in the target actor by mistake. This value was used to scale the actor outputs which are between -1 and 1 to the range of admissible values. This meant the target actor was providing values double that of the actor.

### Returning correct values from memory buffer 🤦‍♂️

This one was the realisation that tipped the algorithm over the edge from not working to finally working. It was also the stupidest thing. On each iteration of the environment you you need to store the state, the next_state, the action that moved the state to the next state and finally the reward obtained. When you run the learning step you take a sample of these recorded values and then update the critic and actor networks. If you accidentally return the state twice instead of the state and the next state then the critic will never learn anything and nothing the actor does makes a difference to the environment. I spent way too long trying to figure out why nothing was being learnt only to discover this was the issue!

## Lesson:

The purpose of listing these errors is to illustrate that non of them threw an exception or gave results that one could easily use to follow to the source of the problem. This is the main difference between RL/ML say something like web dev. In web development mistakes are kind of obvious in that it works or it doesn't. If it doesn't then either an error gets thrown or the app breaks in some reasonably obvious way. In reinforcement learning if something doesn't work then it'll fail but it'll fail in a manner indistinguishable from the way it would fail for most other errors.

I found I had a bias towards assuming mistakes I'd make where more likely to be resultant from misunderstandings of the algorithm theory rather than with my programming. I think this is because the algorithm is inherently opaque. It's hard to say that as a result of a small number of training steps weather the actor subsequently performs better than it did before. As well as this a lot of the logic is hidden behind the [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/) api. Hence I spent a lot of time hypothesising what types of issue might result in the algorithm failing the way it was rather than just looking for the typical set of errors all programmers make. The kinds of things I would have found in a second if this was written into web app or something. I guess the lesson here is write tests for everything!

___


###  Another Error 🤦:

**TLDR**: check numpy operation output and input array shapes are correct!

This error didn't actually prevent the algorithm from working but I thought I'd mention it for those who are interested...

Numpy allows certain operations between arrays of different size. In the case with two arrays of the same size it just multiplies the elements pairwise. If the array sizes don't match then if possible it will "copy" a vector lying along one dimension along the missing dimension/s until the two arrays are the same size. Once this is the case it can then processed with the multiplication operation. "copy" is in quotation marks because it doesn't really copy the array out as that would waste memory.

So as an example:

```py
>> a = np.array([1, 2, 3])
>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>> a*b
array([[ 1,  4,  9],
       [ 4, 10, 18],
       [ 7, 16, 27]])
```
Here the `[1, 2, 3]` vector is broadcast out to become `[[1, 2, 3],[1, 2, 3],[1, 2, 3]]` and then each element is multiplied with it's corresponding one in the other array to get the result. Sometimes the array shapes don't satisfy the rules required to do the copying operation but you can add dimensions in in order to ensure they do. For instance if you have an array `a` that's length `3` and an array `b` that's length `4` then these can't be copied but if you add a dimension to each side alternating so that they go from being `(3),(4)` $$\rightarrow$$ `(3, 1), (1, 4)` then they can:

```py
a = np.array([1, 2, 3])
b = np.array([1, 2, 3, 4])
a[:, None]*b[None, :]
array([[ 1,  2,  3,  4],
       [ 2,  4,  6,  8],
       [ 3,  6,  9, 12]])
```

The process of "copying" along the axis is called [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) and it can be a little hard to think about so when it doesn't throw errors there's a tendency to assume it must be correct.

Consider:

```py
rewards = np.ones((64, 1))
dones = np.ones((64))
```

then what shape is `rewards * dones`? The answer is `(64, 64)` and not as I had been expecting `64`. In fairness `rewards` was actually the predicted rewards and was returned from a Keras model and I'd forgotten they return an array for which the first dimension indexes the batch size. The solution was just to use `dones * rewards[:, 0]` which effectively pops of the last dimension so that the shapes are now both just `64`. This wasn't actually a massive issue and was absorbed later due to [`reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean) which stopped me noticing the shape was larger than I thought it was. At worse it was just more computation than needed!
