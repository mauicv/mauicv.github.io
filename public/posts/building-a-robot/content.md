It's been a dream of mine to build and train a robot to walk. I sort of semi achieved this goal last year when [I trained a quadruped to walk](#/posts/real-world-model-rl). But i wasn't really happy with that project for various reasons. I didn't build the robot myself, the training was messy and didn't really feel very repeatable and I really want 2-legged locomotion. Anyway, I have at least managed to design and build the robot that I want to train to walk, even if i haven't managed to actually get it walking yet! It's been christened Gnoci. I'm pretty pleased with the build itself and learning how to do all this stuff was a lot of fun. I've never used CAD for 3d Modelling or done any PCB design before. It was also a great chance to get into 3d printing as well as electronics. There's something really nice about being able to design embedded systems - however, i'm not an expert and this was my first proper foray into this so inevitably there will be stuff thats wrong.

# Electronics

Everything is based on a Raspberry pi 4 which acts as the main controller. I designed an extension board in KiCAD which breaks out 12 PWM outputs and 16 I2C buses. The PWMs are for 10 servo motors while 10 of the I2C's are used for rotary encoders and 4 for contact sensors. The board also integrates an MPU6050 IMU breakout which Gnoci uses to measure pitch and roll. The 12 PWM's are driven by a 16-channel LED controller (PCA9685 IC) and the 16 I2C busses are provided by 2 8-channel multiplexes (PCA9548A IC).

<table style="width:100%">
  <tr>
    <th>Controller board KiCAD design</th>
  </tr>
  <tr>
    <td><img src='/posts/building-a-robot/kicad-controller-board.png' alt=''></td>
  </tr>
</table>

<!-- ![](/posts/building-a-robot/kicad-controller-board.png) -->

<table style="width:100%">
  <tr>
    <th>Power and Controller board</th>
    <th>Controller board on its own</th>
  </tr>
  <tr>
    <td><img src='/posts/building-a-robot/power-and-control-boards.jpeg' alt=''></td>
    <td><img src='/posts/building-a-robot/controller-board.jpeg' alt=''></td>
  </tr>
</table>


The 10 servos are driven by a 2200mAh 2s lipo battery. This provides 7.4 volts and so we need to drop this down to 5Vs for the pi, so I have a separate power board with a buck converter that does this. Originally the plan was to house everything on a single board but with 12 pwm 3-pin outputs and 16 i2c 4-pin connectors there was not a lot of space (More on this later). As a result there are three power lines, GND, 5V and 7.4V - the 7.4V rail is routed to the servos and a couple of large caps provide capacity if the servos suddenly draw a lot of current. Everything else is powered by the 5V line. 

I made some pretty bad design choices early on in the process which I will change if there are later iterations of this model. The main one was designing everything around the assumption that i'd be reusing the servo motors from the quadruped. I chalk this up to inexperience but I assumed they'd be strong enough to hold the weight of the robot I built (At this point i was aiming to build a much smaller robot). Because I was reusing these servos I'd need separate rotary encoders to measure position. I guess this was done for the purposes of money saving, regardless, it was a bad choice. Once I started actually designing the robot itself it became clear that it would be next to impossible to make it light enough for the servos I was using but by that point a lot of these decisions had been baked in and it was too late to fix wihtout serious effort. In particular I ended up using higher torque Digital Servos that still used PWM to be compatible with the board I'd designed - this meant separate connections for each servo plus each rotary encoder. In retrospect I wish I'd known about SPI and servos with built in rotary encoders. Daisy chaining 10 servo motors with inbuilt sensors is much cleaner than having to have a separate line for each and a single SPI output would have made designing and manufacturing the controller board so much easier! For an example of some guys who obviously know what they're doing take a look at the [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) project. I don't feel too bad about this though, part of the thing with learning new stuff, especially in complex areas is that at some point you just have to commit to a design choice and learn from the outcome.

A slightly odd design feature of the robot is the contact sensor system which is made up of 4 analogue to digital converter ICs connected to limit switches placed at the front and back of each foot. Originally I had intended to use pressure sensors which would mean decoding an analogue value rather than an on/off state but when I saw the [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) project had managed to train walking bipedal robots with just limit switches I decided it would be a good idea to copy them however at that point the i2c design choice in the controller board was baked in so I'm using 16-Bits of resolution to measure 1 bit of information. You live and learn. 

The final issue which I thought had really screwed me but actually hadn't was placing all the peripherals on a single i2c bus... This is also just inexperience. The raspberry pi has three separate i2c pin pairs and at the time I designed the controller board I unthinkingly placed the 2 PCA9548A ICs, the PCA9685 ICs and the IMU IC all on the same bus. The software process for controlling the robot basically requires making 15 sensor read operations, a neural network prediction, followed by 10 motor write operations on this single I2C bus... I had been aiming for 100Hz for the full control cycle but reading through the i2c multiplexes requires two I2C ops. One i2c write to the multiplex to select a channel and one i2c read to the sensor to collect the data. As a result we're talking 14*2 sensor i2c ops + 1 for the IMU op. Because all of this happens on a single bus we can't multithread the work. Fortunately, the motor writes are very fast because the PCA9685 allows you to write multiple registers in a single i2c operation. As well as this the raspberry pi default clock rate for i2c is slower than it has to be - in particular it defaults to 100kHz for maximum compatibility. At 100kHz the control loop ran at 57Hz but by bumping the i2c clock rate to 400kHz we hit 120Hz, more than enough for what we need.

Originally my plan was to use dev boards for the rotary encoders but they proved quite annoying to integrate into the robot joints due to size and lack of clean connectors and so in the end I designed my own smaller footprint boards. In addition, I designed a PCB for the A2D IC's that would measure the force-sensitive resistors that I was going to use for contact detection. Replacing the FSR with a limit switch was trivial.

<table style="width:100%">
  <tr>
    <th>Magnetic Rotary encoder</th>
    <th>Analogue to digital chip and pressure sensor (Force-sensitive resistor)</th>
  </tr>
  <tr>
    <td><img src='/posts/building-a-robot/rotary-encoder-ic.jpeg' alt=''></td>
    <td><img src='/posts/building-a-robot/a2d-ic.jpeg' alt=''></td>
  </tr>
</table>

In general, despite making the above mistakes, everything seems to work and I've not had any major issues which honestly surprised me since this is my first real foray into electronics. Obviously, I'd rather not have as many wires everywhere but ultimately this is cosmetic and i'll clean it up in later iterations.

# CAD design

Initially I learnt FreeCAD but once parts got complex performance suffered enough that I switched to onshape. There's not much to say here except that I made my life much more complex by not using daisychainable servos with built in rotary encoders. Most of the difficulty was making sure each joint had space on either side for the servo and the sensor and this also complicated assembly. The robot has 10 degrees of freedom total, 5 on each leg. The head unit contains the raspberry pi, powerboard, lipo battery, a voltmeter, a fan and a power switch.  

<table style="width:100%">
  <tr>
    <td><img src='/posts/building-a-robot/gnoci-cad-drawing.png' alt='' style="width:100%"></td>
    <td><img src='/posts/building-a-robot/gnoci-cad-design.png' alt='' style="width:60%"></td>
  </tr>
</table>

<table style="width:100%">
  <tr>
    <th>Mid build</th>
  </tr>
  <tr>
    <td><img src='/posts/building-a-robot/gnoci-build.jpeg' alt=''></td>
  </tr>
</table>


Construction was a lot of fun, learning to solder has probably been my favourite part of this process. I'm using surface mount components and i wanted clean looking boards so i opted to use a hot plate to solder the components, after which i cleaned up any bridges with an iron as well as soldered the through-hole components. First attempts were messy but I was happy with the end result.


<table style="width:100%">
  <tr>
    <td><img src='/posts/building-a-robot/hotplate.gif' alt=''></td>
  </tr>
</table>

# Gnoci-sim and RL training

Because I ultimately want to train this using RL I need to build a simulation environment in which to do so. I used [onshape-to-robot](https://onshape-to-robot.readthedocs.io/en/latest/) to download the assembled cad design and map it to a mujoco xml specification. Training the policy itself is ultimately the part that's ongoing and most frustrating. I've managed to get reasonably good walking controllers in sim using an algorithm close to something called TCRL introduced in [Simplified Temporal Consistency Reinforcement Learning](https://proceedings.mlr.press/v202/zhao23k/zhao23k.pdf) as well as [adversarial motion priors](https://arxiv.org/abs/2104.02180).


<table style="width:100%">
  <tr>
    <td><video src='/posts/building-a-robot/TCRL-AMP.mp4' alt='' controls ></td>
  </tr>
</table>


However i've been finding this process incredibly finicky, and while I've had some amount of success getting it to work in sim, i've really struggled transferring to the robot. Partly i'm making this much more difficult for myself because I want to use world model RL instead of model-free methods like PPO which are the more commonly used solutions. Anyway, this is where i'm at with this.

<table style="width:100%">
  <tr>
    <td><img src='/posts/building-a-robot/gnoci.jpeg' style="width:60%; height:60%;" alt=''></td>
  </tr>
</table>

Hopefully soon i'll have an update in which I can show the robot walking (fingers crossed).