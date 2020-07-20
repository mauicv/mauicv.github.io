---
layout: post
title:  "First Impressions of Reinforcement Learning"
date:   2020-05-25 18:37:37 +0100
categories: reinforcement-learning
---
<sup>__note__: *Relevant code for this post is [here](https://github.com/mauicv/openai-gym-solns)*</sup>

___

## First Impressions


So I've receintly started looking at Reinforcement learning in my spare time because it just seems like a pretty powerful tool. I feel like I sort of threw myself in the deep end in that I have no experience with any of the usual machine learning frameworks and while rougthly I understand the underlying theory behind it all I was pretty rusty and Reinforcement learning turned out to be a slight departure from what I already knew.

These blog posts are mostly going to be a summary of everything I've learnt and also bits and pieces of stuff I don't understand yet. I'll start with general takeaways, then go into theory and then give examples of implementations written in tensorflow.

___

## General Problem


All Reinforcement Learning takes place in some environment. An environment is really just a collection of states. Floundering around in this environment is an actor or agent who is at any point in time in a single state of the environment. The actor interacts with the environment by making actions. These actions transistion the actor from the state it's in to a new state. We want to train the actor to move to a particular state or collection of states that fufils some goal of ours.

As an example in the [openai luner lander gym problem](https://gym.openai.com/envs/LunarLander-v2/) the environment is the moon surface, the landing site and the position of the shuttle. The actor is the spaceship and the actions it can take are the firing of each of it's engines or the choice not to do anything. We want to train this actor to succesfully move between states in the environment so as to land itself at the landing location.

There's a couple of approaches to solving this kind of problem. The one I was primarily interested in and spent the most time looking at where policy gradient methods. In policy gradient methods you essentally have a policy that tells you how your agent is going to behave given any state in the system. Initially this policy is random and useless but by using the policy and keeping those actions that it suggests resulting in good outcomes and throwing those that result in bad outcomes you slowy improve the polciy until the set goal is achieved.

The way this works is essentally by makeing the policy a parameterised neural network that takes the states of the envrionment and gives as output a probabailty distrubution describing the actions liklyhood of having a good result. If you sample an action from this policy and it goes well then you compute the derivative of the policy density function at the sampled action and then use gradient acsent to make that action more likley. And vis versa if it goes badly you do the opposite. I'll explain this in greater detail in part 2 so don't worry about understanding it here.

Usually you don't actaully know if a given action was good or bad until later. This is really the crux of the whole thing, becuase maybe something the agent did at a time $$t_{1}$$ made a significant difference to the outcome that results at time $$t_{2}$$. If the time interval $$t_{2} - t_{1}$$ is big then it's not clear that there should be any relationship between the action at $$t_{1}$$ and the outcome at $$t_{2}$$. Theres ways around this in that you can try and shape the rewards allocated through out the training. If you want an agent to learn to walk then maybe its good to reward standing as an action first. But this becomes messy becuase it's hard to define what behavours to reward in between the random behavour and the end goal. Not just this its also pretty labour intensive.

___

## Main obsticles

The major barriors to my personal progress in this domain have defiently been

1. Bits and peices of ungrepable domain specific knowledge
2. Not knowing what to expect from Reinforcement algorithm performance
3. Not knowing how to approach debuging Reinforcement algorithms

#### Domain specific knowledge

Both rl and software developement suffer from weird bits and peices of domain specific knowledge. This kind of thing just comes with the territory. Unknown unknowns undoubtly exist in any feild. The kind of thing I'm talking about here is stuff that's hard to search for becuase you don't know what's going wrong. An example is not knowing that you would typically normalize inputs before feeding them into a nerual network. If not knowing and thus not doing this cuases your training to fail it's not something your going to know to change, beucase, well you don't know to do so, at least not until you somehow stumble across it while searching around on the internet...

#### Expecations and Debugging

In software development we get error messages when stuffs broken. In contrast in rl you get nothing and instead are left to form vauge hypothises about why it's not doing what you want it to do. This it the aspect of the whole process which is perhaps the most frustraighting. Instead of having a clear obsitcel to navigate you have a collection of possible issues. Even worse sometimes there isn't even an issue and what you think is broken is actaully just slow, or your logging the wrong thing. My feeling is there is a kind of werid intuitional side to reinforcement learning where you eventually learn to pick up the subtle indications of each different type of issue, and know the types of things that might solve that issue. It's almost as if through an iterative process we're learning to keep what works and discard what doesn't...

I don't have a great deal of experience of machine learning in general but I have a feeling that rl departs from ml in the sense that it's hard to assertain when something is learning. In ml you typically have a loss function and the model updates ensure that improvements are montonic. So while it may not be improving fast and it may not be converging to a global optimum you do know wether or not it is improving. In reinforcement learning you get something completely different. Sometimes you get this:


![progress](/assets/intro-to-rl/clear-progress.png){:height="50%" width="80%"}


And then sometimes you get this:


![progress?](/assets/intro-to-rl/unclear-progress.png){:height="50%" width="80%"}


The inherent variance in policy gradient methods seems to be a function of two things (or a couple of very vauge hypothises):
 - In machine learning terms, the states are inputs and the rewards are the training data labels. The problem we have within the paradime of rl is that it's typically unclear how the rewards need to be allocated. so the labels are moving targets and we have to trust that in sampling the system enough they'll converge to there true values.
 - An update in parameter space, if too big, can overshoot it's mark and push the network into a inoptimal state. In this inoptimal state the actor will move along a safe orbit until the overshot update will perturb it off this orbit. If it does this at a particularly inopertune moment then i'd guess you can get some kind of divergence of trajectories in which orbits that once went to safe areas of the state space are now redirected into low performing areas. I think becuase there are delays between action and outcome and becuase we discount earlier actions significance w.r.t. the end result it can take a long time for the network to realize that that particular action at that particular time was a mistake. Instead the actor spends ages trying to navigate the poor environment it's been redirected into.

It was this that probabaly led me to bang my head against a wall the most! Often times you watch the performance of the actor improve and improve until it's doing really well until suddenly it plumments to perform worse than when it started. In retrospect this isn't always a particularly bad thing in that in being redirected there the actor will learn how to navigate said poor environment. Building in redundency like this should result in stable solutions that apply under pertubation.

I suspect these behavours are highly dependent on the nature of the reward environment. In the cases where rewards are continuously allocated I'd anticipate reasonably conintuous learning profilse whereas sparse discrete rewards would likely cuase the trajectories to jump between stratigies and thus you'd expect to see a greater deal of variance in learning.

These issues led me to take a kind of messy stop and start approach to the training of the luner lander solution. I'd save the model during training when it was doing well and then revert to the save point if it then flatlined. This may have been more to do with my pschology than any sensible stratigy but over time the model did improve even if that apporach was disapointingly messy.

#### Finding Reinforcement problems

Once I'd started to see progress on the above problems I begain trying to think of areas in which these ideas could be applied. Truthfully it's not as easy as you'd think. In principle anything can become a problem that has a reward on solving but typically the ones you might think of tend to fall foul of the fact that it's hard to run enough real world experiments to see any significant learning. Nasa can only build so many moon landers. It seems like rl is well developed to solve problems that we can simulate on a machine. This is kind of an issue if you want real world applications in that you can train something in a simulation but theres no garunetee the solution will have any cross over and on top of this correctly modeling problem spaces inside physics engines is hardly tirvial.

Another issue that potentially arrises within lots of problems is just the sparcity of rewards. You might try to train an RL to learn to write simple programs. After all there is a task, writing the code, and then there is the reward, getting the correct result on running the code. You can think of this as like a maze, the machine has to choose the exact correct sequence of letters from the alphabet to get to the exact correct string that correctly implements the solution. In this case unless you can find a way of shaping the rewards correctly the model will just endlessly try random combinations of letters. Presumably Human beings can do this becuase they've been rewarded from hundreds of smaller steps along the way so as to arrive at the point where we have all the tools needed to write, debug and correct software. If you want to teach an AI to code you'll probabaly have to teach it all the stuff inbetween too.

___

## Main takeaways

RL is very interesting, both in theory and application, and it seems to be pretty powerful in the sense that I can see there being problems that it can solve that would be very hard if not impossible to code solutions for. I've defienetly been way more negative in the above than positive but i'm pretty excited by this stuff! It's pretty remarkable that these methods work and you get solutions that seem somehow natural.

However it's also been a little disapointing too. This is partly becuase the intial learning curve seemed to be a lot steeper than I expected and also becuase my expectations in general where way higher than they should have been. I didn't try but i suspect in the time it took me to obtain a solution to the luner-lander environment through training I could have easily coded a progromatic solution and with way less of me hitting my head of a wall. And I think this is the major con in that whenever you set out to solve an problem using RL your kind of making a bet that it's going to be possible. There are alot of unknowns and they don't seem to be predictable in any particular way. Like maybe it just happens to be the case that the way the reward envrionment has to be shapped your very unlikly to ever have the model get to the point where it's making progress. It's also just hard to assertain progress. My feeling is that if you can write a progromatic solution to the problem then you should probabaly always prioritise doing so. RL becomes super interesting when we find ourself in the domain of problems that cannot be solved by hand.

The most enjoyable aspect of the whole thing so far has definetly been watching the way that once the luner lander has hovered down to the final 5 pixels it then drops onto the moons surface in a weirdly human way. As if theres actually some guy inside who just flips the engine off switch to bring the vehicle down to ground*.

<sub> * I have no idea how to land aircarft</sub>
