__note__: *Relevant code for this post is [ddpg-ant repo](https://github.com/mauicv/ddpg-ant)

## Intro

So a couple of months ago I had an idea to try and test different types of exploration noise in ddpg environments. The original [ddpg paper](https://arxiv.org/pdf/1509.02971.pdf) suggests using an [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) for exploration. I'd read [elsewhere](https://qr.ae/pG8hlM) that this wasn't actually needed and Gaussian noise would suffice. I wanted to see if using smoother noise helped the critic to learn in a more stable manner. Spoiler alert: It probably doesn't. Note that this is a purely exploratory effort though and not an exhaustive study.

I compared four noise schemes:

<table style="width:100%">
  <tr>
    <th><a href='https://en.wikipedia.org/wiki/Gaussian_noise'>Gaussian</a></th>
    <th><a href='https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process'>Ornstein–Uhlenbeck</a></th>
  </tr>
  <tr>
    <td> <img src='/posts/exploration-noise/normal-noise-2.png' alt='Gaussian'></td>
    <td><img src='/posts/exploration-noise/ou-noise-2.png' alt='Ornstein–Uhlenbeck'></td>
  </tr>
  <tr>
    <th>Linear segment noise</th>
    <th>Smooth segment noise</th>
  </tr>
  <tr>
    <td><img src='/posts/exploration-noise/ls-noise-2.png' alt='Gaussian'></td>
    <td><img src='/posts/exploration-noise/sn-noise-2.png' alt='Ornstein–Uhlenbeck'></td>
  </tr>
</table>

__Linear segment noise__: Selects an ordered set of normally distributed points in the action space and then moves from one to the next in constant increments.

__Smooth segment noise__: Selects an ordered set of normally distributed points in the action space and uses cubic interpolation to generate a function that moves smoothly between each point.

## Setup:

I only considered the [AntBulletEnv-v0](https://pybullet.org/wordpress/) environment in pybullet. I ran two experiments each involving 100 training runs in total. Of those training runs 25 of each was allocated to each noise type. Each training run went for 1000 episodes and for each episode the number of steps was 300. The model being trained had two hidden layers of size 400 and 300. The model architectures are detailed in code [here](https://github.com/mauicv/ddpg-ant/blob/main/src/model.py).

The rest of the ddpg parameters are as follows:

```
LAYERS_DIMS   : [400, 300]
TAU           : 0.001
SIGMA         : 3.0
THETA         : 4.0
BUFFER_SIZE   : 100000
BATCH_SIZE    : 64
DISCOUNT      : 0.99
```

Finally the difference between each experiment was in the learning rates:

Experiment 1 used:

```
ACTOR_LR      : 5e-05
CRITIC_LR     : 0.0005
```

Experiment 2 increased the learning rate by a factor of 10:

```
ACTOR_LR      : 0.0005
CRITIC_LR     : 0.005
```


## Results:

In each case I'm really looking for evidence of one noise process working better than the others. Our sample sizes are tiny because it takes ages and costs to run this stuff remotely. Because our sample sizes are so small I'd really need to see one noise process perform significantly better than the others.

Note that the [AntBulletEnv-v0](https://github.com/bulletphysics/bullet3/blob/93be7e644024e92df13b454a4a0b0fcd02b21b10/examples/pybullet/gym/pybullet_envs/__init__.py#L200) environment should run for a lot longer that 300 steps, namely to 1000. Because of this we're not going to get anywhere close to the 2500.0 required to solve the environment.

### Experiment 1:

The best results achieved for each noise category where 580.0 for the smooth segment noise, 481.12 for the Ornstein–Uhlenbeck process, 470.83 for linear segment noise and 381.04 for the Gaussian noise.


#### Reward density plot

The following indicate the density of reward over all 25 runs at each time step and for each noise process.

![Reward Graphs Experiment 1](/posts/exploration-noise/noise-schema-rewards-exp-1.png)

#### Outcome histogram

The histogram of the 25 end rewards for each noise process:

![Outcome reward histograms Experiment 1](/posts/exploration-noise/hist-plot-mean-rewards-exp-1.png)

### Experiment 2:

The best results achieved for each noise category where 574.47 for the smooth segment noise, 511.55 for the Ornstein–Uhlenbeck process, 595.36 for linear segment noise and 548.44 for the Gaussian noise.

#### Reward density plot

![Reward Graphs Experiment 2](/posts/exploration-noise/noise-schema-rewards-exp-2.png)

#### Outcome histogram

![Outcome reward histograms Experiment 2](/posts/exploration-noise/hist-plot-mean-rewards-exp-2.png)


## Conclusion and issues:

While I think you might be able to make the case that in experiment 1 the smoother noise processes led to slightly better performance I think the small sample size means it's a pretty weak case. Over all I don't think there are significant enough differences between the learning to say anything particularly profound.

While I tried to keep all the variables we weren't interested in constant between runs, it wasn't clear to me how to normalise the variance of each noise processes relative to each other. We'd consider Gaussian noise with different variances as different and so expect different results. Whereas it's hard to say in what way the smooth noise process is the same or different to Gaussian for instance because they're described by different parameters. I wasn't really sure how to account for this so in the end I just ensured each noise fell within the same range of `[-0.02, 0.02]` in each dimension.

## Best solutions:

The best solutions came from the second experiment, in fact in the first experiment both the Gaussian and Ornstein–Uhlenbeck processes basically failed to even get the ant to walk anything further than a couple of steps. Here are the best of each noise category for the second experiment:

<table style="width:100%">
  <tr>
    <th>Smooth segment noise</th>
    <th>Linear segment noise</th>
  </tr>
  <tr>
    <td><video src='/posts/exploration-noise/ssn-exp-2.mp4' alt='' controls></td>
    <td><video src='/posts/exploration-noise/lsn-exp-2.mp4' alt='' controls></td>
  </tr>
  <tr>
    <th>Gaussian noise</th>
    <th>Ornstein–Uhlenbeck</th>
  </tr>
  <tr>
    <td><video src='/posts/exploration-noise/n-exp-2.mp4' alt='' controls></td>
    <td><video src='/posts/exploration-noise/ou-exp-2.mp4' alt='' controls></td>
  </tr>
</table>
