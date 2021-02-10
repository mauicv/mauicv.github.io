---
layout: post
title:  "Quadruped Progress 1"
date:   2021-02-03 00:00:00 +0100
categories: reinforcement-learning
featured_img: assets/quadruped-progress-1/quadruped.gif
show_excerpts: True
excerpt: 'Reinforcement learning and Continuous control systems'

---
<sup>__note__: *Relevant code for this post is [here](https://github.com/mauicv/quadruped)*</sup>

---

<link rel="stylesheet" href="/assets/css/stylesheet.css">

<img
  src="/assets/quadruped-progress-1/ant-walking.gif" alt="ant environment walking solution"
  class="column"
/>
<img
  src="/assets/quadruped-progress-1/standing.gif" alt="drawing"
  class="column"
/>
<img
  src="/assets/quadruped-progress-1/stick-the-landing.gif" alt="drawing"
  class="column"
/>

<br>

I'm feel like I'm starting getting pretty good results from [DDPG]({% post_url 2020-12-23-deep-deterministic-policy-gradients %}) applied to reasonably complex control problems such as standing and walking quadrupeds in simulated 3d environments. I'm modelling the quadruped using [pybullet](https://pybullet.org/wordpress/) and training everything on a [paperspace](https://www.paperspace.com/) machine. It's taking around about a day to get rough solutions to the control problem specified. Although It took me a lot longer to find the correct hyper parameters to start getting results. The nice thing about reinforcement learning is once you get one result you can usually pretty easily get results for problems close by.

So far I've trained it to solve two environments. The first just requires it to stand in place and not fall over the second requires it to move forward. In each it's rewarded for keeping it's body oriented correctly, namely level and pointing in the right direction, and it's punished if it's torso touches the ground. I've also added a cost to applying torque to the revolute joints in order to try and ensure it moves smoothly rather than erratically.

There are still some issues but I'm confident enough now that I've started setting up the hardware. I'm testing out using an Arduino microcontroller which I think will suit my purposes. I built a robot arm test setup in order to get an understanding of how it all fits together. I'm planning to use 16 servo motors, a 16 channel servo driver, 5 Accelerometer Gyroscope Sensors (maybe more) and an Arduino microcontroller all powered with 4AAA batteries. I honestly have no idea if that'll be enough so I'm anticipating there'll be multiple attempts and adjustments.

<img src="/assets/quadruped-progress-1/stuff.jpeg" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; margin-top: 20px;"/>

Initially my plan was to try and build the frame out of thick cardboard and glue as I don't really have room for a 3d printer. This actually works way better than I was expecting but I want the quadrupeds legs to be able abduct from the body and this is proving tricky to do with the materials I have currently. Hence I think I'm probably going to end up printing it all. I'm using blender to generate the stl files which makes a nice break from all the code.

<img src="/assets/quadruped-progress-1/blender-quadruped-progress.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; margin-top: 20px;"/>

The main issue with all of this is the question of to what degree a control system trained in a computer simulation will apply to the real world on real hardware. I figure we cross that bridge when we come to it.
