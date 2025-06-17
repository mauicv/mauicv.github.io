In a previous blog post I talked about [model-based RL](#/posts/rl-world-model). I introduced the idea of learning a model of the world in which to train an agent. In this post I'll talk about how I used this method to teach a quadruped robot to walk.

First off, this has actually been a goal of mine for ages. You've probably all seen the videos coming out of [Boston Dynamics](https://bostondynamics.com/blog/starting-on-the-right-foot-with-reinforcement-learning/) and [Unitree](https://www.youtube.com/watch?v=8ClYBtfhkaw) of robots controlled with agents trained using RL. Ever since I did, I've wanted to work on these sorts of things. I think what really excited me was the idea of being able to build and train our own personal robotic systems, and this project has been a huge step towards that goal for me. I was also very excited when I saw the [Day Dreamer](https://danijar.com/project/daydreamer/) paper come out, and I wanted to try and replicate it with cheaper hardware as well as the transformer world model instead of the recurrent one used in the paper (For what it's worth, I don't think anyone has done this before at time of writing?). I knew this was going to be hard, but I didn't realize exactly how difficult it would be. This was for a myriad of reasons that I'll go into in this post. Please note that this is largely written to mark the progress I've made, so will be a bit light on technical details. Before we start, I'll give a quick overview of the project.

So I bought a quadruped [PuppyPi](https://www.hiwonder.com/products/puppypi?variant=40213129003095&srsltid=AfmBOoqePMjjC4ZpUgJz2nyGiMFGs51nWUm77y5_x-B-iFCvjD7OyRbP) robot (affectionately named Pogo by my sister) developed by a Chinese company called HiWonder. It was the most affordable option I could find that had what I estimated to be the bare minimum requirements for the project. It's a great piece of hardware; however, being on the cheaper end of the spectrum, it was missing some features that in retrospect I should have considered. More on this later. All in all, this project probably cost around £1000, which includes the robot, the Raspberry Pi and camera, and GPU compute. The main uncertainty was the cost of the compute as you don't know how many attempts you'll need to get a policy that works. If I'd known the hyperparameters and caught all the errors before starting, then I think the whole project would have been closer to £700. However, I suspect using something like a Unitree robot would have made the project a lot easier.

The challenge was to train Pogo to walk; in particular, I wanted Pogo to be able to "walk" about a meter in a "straight" line without falling over. If a meter doesn't seem like much, note that Pogo is quite small and I live in London where floor space is at a premium. "Walk" and "straight" are in quotes because, to be honest, I'm willing to be quite lax about exactly what that means.

My understanding is that the current industry approach to most of this stuff is to use model-free RL. In this case, you're training the agent in a simulated environment instead of the real world. This clearly works as you can see in the videos, but it's less exciting to me for a couple of reasons. Firstly, it's likely not how humans learn motor control, and secondly, it's less generalizable to different environments. Part of the power of this kind of model-based RL is that you're not limited by the extent of the simulations you can create. This being said, in practice, learning a model of the environment is very non-trivial. If you're looking for results and it's easy to simulate your problem, then model-free RL is definitely a better option.

## Training Pipeline

The training pipeline I set up consists of 5 main components. The first is the Colab notebook that runs the training loop. This is roughly the same code as I used to solve the simulated environment in the [model-based RL](#/posts/rl-world-model) post. The key difference is that the model is now trained on data collected from the real world. This data can't be generated in the notebook from a simulation like before. Instead, it needs to be gathered in a separate script running locally on my machine that samples rollouts from Pogo and uploads them to a Google Cloud bucket. The Colab process can then fetch the uploaded rollouts asynchronously in between training epochs. Not only this, but the Colab notebook also contains the actor model in the process of being trained, which we need to use locally to collect the rollouts. Hence, the Google bucket acts as an exchange for the actor and the data. The local script can download updated actors and use them to sample from the environment and then upload the rollouts after it has done so.

The local script is responsible for sampling rollouts. It interfaces with the two peripherals: Pogo and the camera. Essentially, it can read state from both the camera and Pogo, pass that state to the actor model (Pogo's brain), and then issue motor commands to Pogo telling him how to move. It does this in a loop until the rollout is over.

![](/posts/real-world-model-rl/pogo-training-pipeline-2.png)

## State and Action Space

In [this blog post](#/posts/rl-world-model), the world model is trained on images of the simulation. This is really useful as it allows you to generate an imagined rollout and then look to see if it's doing something that makes sense. However, the simulated environment is much easier to render useful images of. In particular, the camera can be set to center on the agent and follow it as they move. In contrast, this is much harder to do in the real world, but not only this, learning from the images means extra steps in terms of processing as well as encoding the image data as it comes in. In the simulation, we can take as long as we need to do this. But in reality, where we're learning real-time control, we want to be able to compute actions at 10-20Hz. This is certainly possible, but it's more involved. Hence, I chose to learn from measured states of the robot instead, which are much more compact and thus quick to process or transfer.

Pogo has an MPU5060, which is a 6-axis accelerometer and gyroscope. I augment this with a complementary filter to get an estimate of roll and pitch. As well as this, Pogo has 8 servo motors. The actor outputs angle deltas which are sent to Pogo and update the setpoints of the PID processes which control each motor. These setpoint values as well as the PID update values are all appended to the state vector to give a 24-dimensional state space. Note that the action space is 8-dimensional, one for each motor. I use a Butterworth filter to smooth the actions.

Initially, I naively thought that I could use the accelerometer to compute the reward function. It measures acceleration, so integrating would give velocity which I could then use to compute a forward velocity reward. Integrating the acceleration like this is very inaccurate as any error in the acceleration measurement is accumulated. As a result, I needed to find a better method to measure the velocity. Pogo has a camera, so I tried visual odometry first. This was very unstable due to Pogo's erratic movements as well as camera resolution issues. It might be possible to make it work, but I thought the various constraints would make it too difficult. I next tried placing an ArUco marker on the wall and having Pogo compute the distance to the marker. This worked a lot better but also had issues. For instance, it required Pogo always having the marker in view, which wasn't guaranteed, and the missed detection rate was still quite high due to the low resolution of images and erratic movements. In the end, I used a separate Raspberry Pi with a camera which I mounted on my wall overlooking a portion of the room that Pogo was learning in. I then placed ArUco markers on Pogo himself as well as on the floor at the point I wanted him to walk to. This worked really well.

<table style="width:100%">
  <tr>
    <td> <img src='/posts/real-world-model-rl/big-brother.jpeg' alt='Big brother'></td>
    <td><img src='/posts/real-world-model-rl/pogo-and-friend.jpeg' alt='Pogo and friend'></td>
  </tr>
</table>

The above setup allowed me to measure the distance between Pogo and the target which I pass through some filters to smooth the signal and from this I compute the reward.

## Training

The process of training involved starting the Colab notebook and then intermittently performing rollouts on Pogo. This was the most frustrating aspect of the whole process. A disadvantage of Pogo is that he's not designed to be able to right himself if he falls over. This means lots of stooping down to pick him up and putting him back on his feet. He also has a battery life of about 30 minutes, so he needed to be charged up between each session. Not only this, but it's very hard not to anthropomorphize Pogo (probably shouldn't have named him) and get annoyed when he stubbornly falls over for the millionth time. I've spent a lot of time feeling like a bad parent recently.

The main difficulties here were:

__Reward function design__ The reward function is the thing we're trying to maximize. This was one of my blind spots. I'd solved lots of simulated environments with predefined reward functions, but I haven't designed one myself. In the end, I treated this a little like a hyperparameter to tune and ran lots of experiments with different reward functions. I'd then come up with hypotheses as to why one did better than another and select/adjust and repeat. In the end, I settled on three main terms:

1. $\circ$ A positive reward for closeness to a specific standing posture.
2. $\circ$ A velocity reward that encourages forward movement.
3. $\circ$ A penalty for falling over.

In particular, in order for him to collect the velocity reward, he needed to be within a certain proximity of the standing posture. This is to prevent him from learning to crawl.

__No proprioceptive sensors__ Pogo has no way of telling certain things about his posture. He doesn't have foot contact sensors, so he can't tell if he's touching the ground, nor do the servo motors have feedback on their position. This means that if a servo experiences resistance from the floor, Pogo can't rely on his internal PID model of that motor's position to tell him what's going on. It's possible that the world model might learn to infer this from other sensors such as the IMU, but I think this is likely hard.

__Sensor noise__ All of the sensors are noisy! The camera is noisy, and the IMU is noisy, and there's no way this doesn't affect the world model.

__Timing__ All of Pogo's code is written in Python, which isn't exactly a performance-focused language, and the rollout loop runs at about 10Hz, which isn't the fastest. This is half the speed they use in the [DayDreamer paper](https://arxiv.org/pdf/2206.14176).

__Method complexity__ Finally, getting all the pieces to work together is a nightmare. In software development, when something goes wrong, an error message is given. This is great as it typically points at the source of the problem. In deep learning, something can go wrong and you won't get an error message; instead, it'll just not learn or learn badly. RL is like noisy deep learning where you can do most things right and it still just gets stuck or something and fails to learn anything. Now put all these pieces together, all their hyperparameter tuning requirements, along with noisy sensor measurements and coordinating a training pipeline with 5 separate components, and you've got a recipe for a lot of frustration. So often, I'd be trying to solve a problem completely unsure if it was the actual issue I needed to solve.

Fortunately, because the rollouts themselves are only used to train the world model, this means we can run multiple Colab experiments with different hyperparameters and reward functions. I could then use the agents from each of these to gather rollouts and put them all in the same bucket on which the world model is trained. I guess you could take this further and train a single world model in one process and then train different actors in other processes, but I didn't take it that far. This did mean that once I had enough data, I could run multiple experiments at once, which helped speed up the process without wasting any data.

All the above being said, I did end up with a policy that satisfied the requirements. Full disclosure, it was definetly cherry picked from the policies that emerged during training rather than the final result of the training. In general, the stability of the method is something I plan to keep working on. Regardless, it's not by any means the most impressive policy, but it works. I also just think it's cool that I created a small physical mirror of reality and then within it trained a robot to walk.

<table style="width:100%">
  <tr>
    <td> <video src='/posts/real-world-model-rl/rollout-1.mp4' alt='rollout-1' controls></td>
    <td><video src='/posts/real-world-model-rl/rollout-2.mp4' alt='rollout-2' controls></td>
  </tr>
</table>

## Code:

The code for this blog post is quite extensive and spread across multiple places. The key pieces are as follows:

- $\circ$ [Control and sampling code for Pogo and the camera](https://github.com/mauicv/pogo_control)
- $\circ$ [The world model RL repo](https://github.com/mauicv/world-model-rl)
- $\circ$ [My transformers repo (for the transformer world model backend)](https://github.com/mauicv/transformers)
- $\circ$ [The training notebook](https://colab.research.google.com/drive/1DYSs7cjL6v7FgubEhDXVKfIMJdl1XdZU)

These are all in varying states of organization and quality. At some point, I might try to make something more coherent.