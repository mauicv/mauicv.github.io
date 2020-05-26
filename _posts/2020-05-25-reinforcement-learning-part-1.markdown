---
layout: post
title:  "Reinforcement learning part 1"
date:   2020-05-25 18:37:37 +0100
categories: machine-learning
---

---
## First Impressions


So I've receintly started looking at Reinforcement learning in my spare time because it just seems like a pretty powerful tool. I feel like I sort of threw myself in the deep end in that I have no experience with any of the usual machine learning frameworks and while rougthly I understand the underlying theory behind it all I was pretty rusty and Reinforcement learning turned out to be a slight departure from what I already knew.

These blog posts are mostly going to be a summary of everything I've learnt and also bits and pieces of stuff I don't understand yet. I'll start with general takeaways, then go into theory and then give examples of implementations written in tensorflow.


---

## General Problem


All Reinforcement Learning takes place in some environment. An environment is really just a collection of states. Floundering around in this environment is an actor or agent who is at any point in time in a single state of the environment. The actor interacts with the environment by making actions. These actions transistion the actor from the state it's in to a new state. We want to train the actor to move to a particular state or collection of states that fufils some goal of ours.

As an example in the [openai luner lander gym problem](...link) the environment is the moon surface, the landing site and the position of the shuttle. The actor is the spaceship and the actions it can take are the firing of each of it's engines or the choice not to do anything. We want to train this actor to succesfully move between states in the environment so as to land itself at the landing location.

There's a couple of approaches to solving this kind of problem. The one I was primarily interested in and spent the most time looking at where policy gradient methods. In policy gradient methods you essentally have a policy that tells you how your agent is going to behave given any state in the system. Initially this policy is random and useless but by using the policy and keeping those actions that it suggests resulting in good outcomes and throwing those that result in bad outcomes you slowy improve the polciy until the set goal is achieved.

The way this works is essentally by makeing the policy a parameterised neural network that takes the states of the envrionment and gives as output a probabailty distrubution describing the actions liklyhood of having a good result. If you sample an action from this policy and it goes well then you compute the derivative of the policy density function at the sampled action and then use gradient acsent to make that action more likley. And vis versa if it goes badly you do the opposite. I'll explain this in greater detail in part 2 so don't worry about understanding it here.

Usually you don't actaully know if a given action was good or bad until later. This is really the crux of the whole thing, becuase maybe something the agent did at a time $$t_{1}$$ made a significant difference to the outcome that results at time $$t_{2}$$. If the time interval $$t_{2} - t_{1}$$ is big then it's not clear that there should be any relationship between the action at $$t_{1}$$ and the outcome at $$t_{2}$$. Theres ways around this in that you can try and shape the rewards allocated through out the training. If you want an agent to learn to walk then maybe its good to reward standing as an action first. But this becomes messy becuase it's hard to define what behavours to reward in between the random behavour and the end goal. Not just this its also pretty labour intensive.

## Brief Interlude

If you think about it this raises the question of how it is we are rewarded ourselfs. What is it that encourages our behavour? Well obviously furthering our genetic material but in between nothingness and gene propigation theres a tun of intermediate rewards that evolution has placed in order to make this process less of a floundering around in the dark and more a shaped scheme for progressing towards having and rearing equally succesful children. In the case of evolution these middle rewards just emerged from the process of compitition within our environment. Initially we didn't need to walk we just sort of rolled around in goo and ate stuff... and that was enough to get us to the end goal. But a red queen arms race later and somewhere in between hanging around in puddles and now it became useful to learn to walk among other things.

> Many were increasingly of the opinion that they'd all made a big mistake in coming down from the trees in the first place. And some said that even the trees had been a bad move, and that no one should ever have left the oceans.
<sub>The hitchhickers guide to the galaxy, Douglas Adams</sub>

It would be interesting to see if one can generate this cascade of intermediate rewards with the aim of actually creating one that was the primary goal. So rather than start with the intial goal start with something arbitrary and then enforce some kind of compitition between different solns that cuases rewards to emerge independent. You can imagine having two agents that are trying to have as much success in an arbitrary task as possible but also have to have policies that differ by some amount. This means if one gets better than the other then it has to find a way of doing even better but using a different soln, and so on... 

---

## Main obsticles

The major barriors to my personal progress in this domain have defiently been

1. Bits and peices of ungrepable domain specific knowledge
2. Not knowing how to approach debuging Reinforcement algorithms
3. Not knowing what to expect from Reinforcement algorithm performance

Both rl and software developement suffer from weird bits and peices of domain specific knowledge. This kind of thing just comes with the territory. Unknown unknowns undoubtly exist in any feild. The kind of thing I'm talking about here is stuff that's hard to search for becuase you don't know what's going wrong. An example is not knowing that you would typically normalize inputs before feeding them into a nerual network. If not knowing and thus not doing this cuases your training to fail it's not something your going to know to change, beucase, well you don't know to do so, at least not until you somehow stumble across it while searching around on the internet...

In software development we get error messages when stuffs broken in contrast in rl you get vauge hypothises about why it's not doing what you want it to do. Similarly, In software it either works or it doesn't in rl we run an algorithm for 2 hours thinking it's learning only to have it's performance drop of a cliff in the last 20 episodes of training.
