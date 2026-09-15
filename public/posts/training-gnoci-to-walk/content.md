This post follows on from the ["_I built a robot_"](#/posts/building-a-robot) post and in particular talks about my effort to get the robot I built to walk. Similar to the ["_I trained a quadruped to walk_"](#/posts/real-world-model-rl) post, I set out with the goal of just wanting it to walk across a "small" room and technically I succeeded but as usual there is a huge gap between what I was imagining and the actual results. Regardless, as with all projects issues have arisen during progress that have accumulated to the point that I think it makes sense to prioritise a redesign/overhaul/V2 rather than keep trying to get this model to work. I've never done any serious electronics or CAD, etc prior to this project so a lot of early desicsions where mostly about making progress in order to learn what does or doesn't work rather than doing things the "right way". After a while these decisions have caught up with me. Ultimatly this post is an attempt to list what I've learnt, the issues I had and what to change moving forward to V2.

## What's been done

I've trained a collection of policies in simulation using PPO and sucessfuly deployed some of them on a real robot that I designed and built myself. The real robot in question (called gnoci) is made up of a frame designed onshape and printed in PLC. It's controlled by a raspberry pi 4 with a controller board extenstion that breaks out 10 pwm outputs for each of the servo motors, 16 i2c connections for the 10 joint rotary encoders and 4 foot contact sensors as well as an IMU, power board, LIPO battery, fan and voltmeter. All of this is modelled in mujoco and available here: [gnoci-sim](https://github.com/mauicv/gnoci-sim). Finally, the controller for the raspberry pi is here: [gnoci-control](https://github.com/mauicv/gnoci-control). The two best policies resulted in gates that could travel pretty far (definetly across a small room) and with a reasonably low probality of falling over. There where two main gaits that emerged, the first was a slower stride and involved the robot leaning heavily on one leg while swinging the other the second was a much faster gait with many small steps 

<table style="width:100%">
  <tr>
    <td><img src='/posts/training-gnoci-to-walk/sent-hold-fight-sim.gif' alt='' style="width:100%"></td>
    <td><img src='/posts/training-gnoci-to-walk/dinner-great-lasso-sim.gif' alt='' style="width:100%"></td>
  </tr>
  <tr>
    <td><img src='/posts/training-gnoci-to-walk/sent-hold-fight-real.gif' alt='' style="width:100%"></td>
    <td><img src='/posts/training-gnoci-to-walk/dinner-great-lasso-real.gif' alt='' style="width:100%"></td>
  </tr>
</table>

The core critisim of these policies is that they take painfully short steps and this has been my major hangup about calling this initial project done. However I am pleased that the policies have seem to learnt to balance well enough in simulation that it transfers to the real robot. 

At some point I started getting perhiperal read/write errors which would cuase sensor and servo dropouts severe enough that policies that worked previously no longer did. I think this may just be faulty connections arising repeated collisions with the floor or the batteries getting old but regardless I decided this was a sign and Gnoci V1 had taught me enough that I could move onto V2.


## Key steps:

### Sim2Real

Sim2Real referes to getting an accurate enough model of the robot that a policy trained in simulation can be deployed in reality. There are three approaches to solving this problem. You can make the simulation more accurate, you can make the real robot more predicatble and you can make the policy more robust. I did hefty amounts of all three. 

Gnoci was designed in Onshape and then exported to Mujoco xml description file using [onshape-to-robot](https://onshape-to-robot.readthedocs.io/en/latest/). After this its mostly a case of getting good measurements of masses, making sure the sensors and actions are aligned and on the same scale.

In order to model the joints as best as possible I built a test arm rig and performed a set of recorded rollouts. The idea here is you record the effect of actions on a simple system with a single servo and sensor. You also model the simple system in mujoco and then perform some kind of search (I used [NES](#/posts/rl-nes)) over mujoco parameters in order to find the set that minimize the error between real and simulated rollouts. Initially I used chirps and step functions but in later iterations I also used action sequencies taken from policies I'd trained and deployed on the robot. I had a degree of success with this but there was also a fair amount of variance in the parameters that would come out.

<img src='/posts/training-gnoci-to-walk/sysid-rollout.png' alt=''>

<sup>_top plot shows real and simulated sensor readings from the test arm, lower plot is the action_</sup>

I think domain randomisation did most of the work in the end. I randomised a bunch of stuff like, initial posistiomns, mass, floor tilt, friction, joint armature, dampuing, gain, etc. I think the key things here where adding a random IMU bias to each simulated rollout, lots of observation noise and action delay.

During testing of policies I also noticed that the policies themselves tend to give quite noisy action signals which in turn caused a fair amount of chatter and sim2real gap. In the end I mostly solved this by 1. adding a action rate loss to the training 2. a low pass filter in both sim and real that smooths the action signal 3. a maximum joint velocity that acts as a hard cap on commanded difference in posistion.

There where some things I tired but didn't get working properly and never fully figured out before Gnoci retired himself. In particular adding mechanical slack to each joint in the robot. I did this by adding a secondary joint at each degree of freedom with very little friction but only 1 or 2 degrees of range. This idea isn't mine but taken from [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini). This change tended to result in policies with more unusal gaits which often relied on leaning heavily on one leg which wasn't ideal. I'm sure this would have been fruitful to persue further, perhaps with some adjustments to the reward function.

<img src='/posts/training-gnoci-to-walk/joint-slack.gif' alt='' style="width:100%">

### Reward function:

The reward function I ended up with was made up of 10 seperate components.

- __1.__ _stand_: Measure of Height and Upright orientation, not used as a reward directly but gates velocity and posture rewards (velcity, foot-swing, orientation, yoke-joint-posistion). In later iterations using as an independent reward rather than a gate also worked.
- __2.__ _velocity_: Forwards velocity
- __3.__ _rotation_: Penalised for yaw rate taken from IMU sensor. Added to prevent policies that drift to side or walk around in circles.
- __4.__ _strafe_: penalised for sideways motion. Added to prevent policies that drift to side.
- __5.__ _foot-swing_: two reward components: Duration of foot lift and height of foot lift. Added to encourage policies to learn natural forward gaits.  
- __6.__ _orientation_: rewarded for keeping pitch and roll of robot body close to 0. Added for stability.   
- __7.__ _yoke-joint_: Rewarded for keeping the first 2 degrees of freedom in each leg close to 0. These are leg rotation around the body and outwards lift. Without this reward the robot would often learn lopsided gaits.
- __8.__ _action-bounds_: Rewarded for keeping the actions inside allowed joint ranges. This was added to prevent bang bang control policies that where emerging at some point.
- __9.__ _action-rate_: Penalise high rate of change of policy actions. Added to prevent servo chatter and bang bang policies as well as increase policy stability. 

Note that a lot of the above these are perhaps redundant and where added to solve various problems and but kept in once better solutions to those problems came along. I did find that weighting the foot swing reward highly was key to the policy learning a good walking gait without this the policy would often fail to discover that it could lift a foot and catch itself. Later I also tested a Raibert style foot tacking reward and in particular discovered that this could work even in the absence of the forward velocity reward. This is meaningful becuase if I ever get round to the world model rl version of this where we're training in base reality things like forward velocity become a lot harder to measure.

<img src='/posts/training-gnoci-to-walk/riebart-only.gif' alt='' style="width:100%">

### Miscellaneous

- __1.__ PPO uses a stochastic policy and encourages exploration via a hyperparameter eta which increases the entropy of the action distributiuon each step. However, I found using the intial standard deviation of the policy instead of eta a much better approach to controlling exploration. I found eta hard to tune, it either would result in too much exploration or too little. Whereas Increaseing the initial exploration avoided both these issues as the policy reduces the varaince over time anyway. This was a key discovery to getting results. In particular certain simulation changes impact the exploration significantly, for instance changing the actuator parameters. Often I thought a specific version of the simulation was just not possible to train but actually a change in the acutators meant the amount of exploration had been significantly reduced.  
- __2.__ When I trained Pogo, I used absolute posistion actions rather than delta actions. What I mean by this is that the policy can either output the posistion the robot should be in, or a delta value that tells the robot how to change its current posistion. So instead of "put your foot here" its more like "move your foot forward". This as i recall was a key change required to get Pogo to walk, so i had carried it over to Gnoci. However in Gnoci's case it defeinetly turned out to be better to use absolute posistions instead of deltas. I think this is mostly becuase absolute posistions is better from a sim to real perspective which is much more important for biopedal robots that need to balance wheras Pogo was a quadruped that wasn't trained in simulation anyway.
- __3.__ Not really surprisingly Frame and Action stacking where very important, espcecially when modelling action and observation delays.

## Mistakes/Issues

The above is a short summary of how this has resolved but the real story is a litany of mistakes. My number one character flaw is overreaching and this whole project was a reach from the start.

### Hugely overcomplicated things

I wanted to use world model RL instead of the industry standard (model free/immitation learning). World model RL involves a number of complicated algorithims that arn't really adapted into frameworks yet and so you have to write from scratch. It also requires training on real world rollouts, but this comes at the price of the robot inevitably damaging itself after repeatedly falling over (Mostly I wanted to avoid having to repeatedly pick him up whenever he fell over as I had to do with [pogo](#/posts/real-world-model-rl)). I compromised with myself by planning to train it in simulation and then fine-tune in real, so I ended up writing all the simulation code anyway which somewhat voids the purpose of doing world model RL. In the end I had too much difficulty getting a policy to work with the world model RL and even those that did took a painfully long time to train. Eventually, I decided to revert to model-free PPO and return to world model RL in a later iteration. PPO trained much faster, seemed to learn more natural gaits and was much more robust to domain randomisation. This was all further complicated by the fact that I'd decided to build the robot from scratch requireing me to learn CAD and PCB design etc. In a particular I also wanted to use the same motors from [pogo](#/posts/real-world-model-rl) which turned out to be a huge mistake. I herein commit to only ever do one complicated thing at a time from now on. I can learn electronics for a project, or cad or try to do novel research but not all at the same time. 

### The servo Saga

For the record, I knew setting out that my choice of servo motors was risky, I just really weanted to save money by reusing the motors from [pogo](#/posts/real-world-model-rl). These don't have built in rotary encoders, i didn't even know that was a thing at that point, so I needed independent rotary encoders. At the initial conception of this idea this all seemed very doable so I baked this desicion in early and in particular deisgned the controller board PCB with this in mind. I probably should have realsied that breaking out 10 pwm headers and 10 plus I2C outputs was not a great design choice but in my defence I was vaugly replicating what I knew which was pogos controler board. Subsequently, when deisgning the frame of Gnoci in CAD I realised that having independent sensors on each joint meant the joints had to be more involved and thus larger than I initially anticipated all of this meant scaling the size of Gnoci up substantially and as a result the servo motors from Pogo where no longer strong enough for the project I was intending to build. I had to upgrade the servo motors, but becuase I didn't want to redesign the cntroler board PCB I got PWM input servos instead of geting a set of CAN/SPI input servos with built in rotary encoders (that I now knew about). As a result gnoci has way more wires that is sensible, which has lead to nemerous sensor reading issues, wires pulling out, faulty connections, asthetic issues etc.. and the complexity of designing the joints with the sensor offset on the same rotational axis from the servo motor means that each of the joints has a fair amount to mechanical slack, which on top of the servo gear slack adds up to a painful amount of overall flex on the legs.

Servo motors also present an issue when it comes to sim2real. My understanding is that from a modelling standpoint they're alot harder to simulate than ..., the types of servo typically used for this type of project. This was a problem which i was aware of to start with, but my solution was world-model RL which would in theory just learn the servo dynamics directly and I wouldn't have to model them correctly in the sim... this might have been the case but as mentioned above I ended up just using model-free RL which meant I had to model the servos. The main issue I had with the RL was Sim2Real which I get the impression is the norm for this type of project and this wasn't helped by the choice of servos. Not only this, but these types of geared servo motors tend to have very little compliance. Complicance referes to the machanical flexibility or give in a system, an electric motor on its own has high complaince because typically its easy to push back on the motor position, however add a high ratio gear box and the friction and mechanical advantage work to lock the motor posistion decreasing complaince. High complaince is useful for walking robots becuase when a foot strikes the ground that impact is transfered up the leg through the motors. If the motors are complicant they'll soak the impact up, if there not then they're will be a much sharper shock to the robot. I think this also has ramifications for the reinforcement learning of policies. Smooth impacts and smooth responses mean more predictability and more stability, sharp impacts and fast high gain respoinses mean unstable polices. 

### Robot Proportions

I think the proportions of the robot probabaly worked against me. Mostly they where fit to factors like electronic and motor sizing rather than bipedal stability. 

<table style="width:100%">
  <tr>
    <td><img src='/posts/training-gnoci-to-walk/gnoci-sizes.png' alt='' style="width:100%"></td>
    <td><img src='/posts/training-gnoci-to-walk/open-duck-mini-sizes.png' alt='' style="width:100%"></td>
  </tr>
</table>

The above shows the proptional differences between Gnoci and [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini). In particular:

### Foot Size

Gnocis small feet in proption to the size of his legs make balance significantly more challenging. You can see from the gates that gnoci's pitch is reasonably unstable and it's having to do most of its work balancing which eats into its step size. In later iterations I changed this and gave him big ski like feet in an attempt to make him more stable. I never got a chance to test this properly on the robot however. 

### Gnoci's center of mass is lower than Duck Mini's.

You can model a walking robot as a inverted pendulum with the pendlum connected to the floor and the Mass suspended at the top corresponding to the robot COM. Any offset from the stable point will result in a component of the resultant force to point horizontally and this results in a torque of $$\tau = mgR\sin(\theta)$$. The moment of inertia of the COM, $$M$$ at a distance $$r$$ is $$I=M r^2$$ and so because $$\tau = \ddot{\theta} I$$ the angular acceleration of the COM is $$\ddot{\theta}=\frac{g}{R}\sin(\theta)$$. Thus increasing the height of the center of mass results in slower tipping and more time for the robot to correct. This is the same reason its easier to balance a meter stick than a 20cm ruler.

### Gnoci's legs are a much larger fraction of overall robot mass than Duck Mini's. 

This issue here is that in order to walk or correct instablity a robot has to move its legs and if those legs have a lot of mass proportional to the rest of its body the resulting intera of that mass moving will transfer to the entire body decreasing stability. If the legs are lighter then they're easier to move and control.

### Gnoci's hip width is much larger!

This results in policies that sway a lot more than desirable. When your walking and you want to lift a foot you need to move your COM over your other foot in order to unload the swing foot. You have to do this for long enough that you can meaningfully move the unloaded foot. For wider hip width robots this means moving more mass more forcfully in order to get the required time to stride the other foot. Narrower hip width however is much easier. You can see this effect directly in some of the polices trained

<table style="width:100%">
  <tr>
    <td><img src='/posts/training-gnoci-to-walk/high-swing-rollout.gif' alt='' style="width:100%"></td>
    <td><img src='/posts/training-gnoci-to-walk/high-swing-rollout-2.gif' alt='' style="width:100%"></td>
  </tr>
</table>

These kinds of policies would often fail becuase the lean would place a lot of stress on the hip joint and that would move the rotary encoder magnets enough to get erronous readings which in turn would destablise the policy. A lot of the reward tuning was trying to tone down the degree of sway.

## Conclusion

I think if I where doing a smaller next revison then fixing gnoci's propotions would be the obvious next step. In manyways I don't think doing so is too big a fix. Increasing the leg length and giving him bigger feet is simple. Decresing his hip width and leg mass is slightly tricker but possible. For instance I could reduced the number of servos used, in particular with three, two at the hip for Flexion and Abduction movements, and one at the knee for knee Flexion movement. With these 6 degrees of freedom you can still in theory train walking policies that can turn using asymetric gaits. Removing motors would significantly reduce the Leg mass and would also make it easier to design narrower hips. I would do this but mostly I'm just really annoyed about all the wires dangling around and I really want V2 to be cleaner w.r.t. this. This ultimatly means using motors with integrated rotary encoders and serial bus interfaces. I may even shell out for quasidrive motors. I'm also going to use [Mujoco playground](https://playground.mujoco.org/) or [Issac-lab](https://github.com/isaac-sim/IsaacLab) instead of writing everything from scratch. 

I also want to keep working on the world model RL approach however. The main thing here is actually robot design. There are two big things I think would be needed, firstly learning in base reality means falling over a lot which often results in damage to the robot/motors etc. So ideally any robot I build would have to be small and light enough to survive repeated impacts. Secondly I'd want the robot to be able to stand up on it's own if it falls over. This would mostly be for my sanity as repeatedly picking up toppled robots is pretty soul destroying.