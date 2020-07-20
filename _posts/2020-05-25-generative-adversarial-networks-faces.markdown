---
layout: post
title:  "Machine Learning, Computer Generated Faces"
date:   2020-05-25 18:37:37 +0100
categories: machine-learning
---

___

## Motivation


So I initally got into generative models becuase I was interested in the idea of planning within reinforcement learning. Currently all the reinforcement learning algorithims I've built focus on what the immidate action to take at a set point in time should be. Whereas I was wondering about setting out larger plans of how to navigate environments to obtain rewards. So suppose an actor finding itself at a particular point in the environment reacts by taking a specific action such as lifting a leg becuase previsouly all the times it lifted a leg went well. This is how policy gradient methods work whereby they take an environment and output a sinlge action. Instead imagine the actor in finding itself at a specific point in the space thinks okay i'm going to keep walking in this direction and then turn right over there. In this case instead of a single action it's conceptulized an entire set of actions that are aimed at obtaining rewards.

The benifit to this apporach is that it potentially allows the actor to segment behavours into different types and resolutions. For isntance if an actors learnt to walk and is trying to get from A -> B but in the interem of doing so falls over. By understanding that there are seperate componenets to the set of actions they're taking, namely the set of motions required to walk and the specific lcoations they're walking to, the actor can focus on specific aspects of the approach that failed them.  So in the above case they fell over rather than got lost so perhaps they need to focus on learning to walk rather than navigating. Of course this also means understanding that there are seperate failure modes. Anyway the above is completly unrelated (and untried) to everything else here but rather it's the inital motivation.

I started by looking at trying to get computer generate imgs of doodles i'd drawn. I have this giant doodle i'd been noodling away on for a while and by taking a set of pictures of it and then subsampling bits of it I hoped that this would be enough to generate new doodles in the same style. I got a little lost. Remember i'm new to all this stuff and the main thing I think i'm comming up against is not really knowing what  to expect from the process of computer learning. Like how long before you give up on a model learning something, how a model can fail, what it looks like for a model to fail and so on. I'm intending on coming back to the doodle example but in order to better understand what was going on I decided to opt for a dataset which better benchmarks, namely faces. The sole aim here is to end up with a computer generated picture of a face that is at least vaguely believably human.

___

## Naive Approach

So the first question I had was why can you not just take a model input a binary 0 or 1 and say if 0 produce this style of face and if 1 produce a different style of face. This is like reversing the inputs and outputs. Instead of getting labels out of a network and comparing there difference from what's expected you instead get images out and ask how far is the image from it's label image. You can do this and you get this ghostly pair.

![happy-and-sad](/assets/generating-faces/happy-and-sad.png)

The main thing to note here is that it's obviousely generating an image that's in some sense an average of all smiley faces or non-smiley faces rather than a specific instance of a smiley face. This makes sense becuase there is only one input either 0 or 1 and it's expected to map each of these values into an image that best represents a wide range of pictures of faces. If it where to create a face that is very face like then it may be close to some faces within that set of faces but it will nessiserily also be far away from others. So instead of a instance of an identifiably human face you instead get a eary blurred mask that represents lots of faces all at once.

So the I kind of expected this would happen. The issue is that the inputs don't give any room for manouver. Your asking these two values to describe the entireity of the dataset so of course the output will be a blended version of all the images. The natural solution to this is to make the space that represents the faces larger. There are some obvious difficulties to navigate here. Suppose you had a better labelling system. So instead of 0 or 1 lets make it a continuum. In the above exmaple 0 was not-happy and 1 was happy. So with a continuum we could represent a sliding scale of happiness where some faces are more happy than other faces. Then we can add  other dimensions. So instead of just a happiness dimension we can have a hair dimension and a nose size dimension and so on... If your willing to spend the time going through the data set and labelling each image by where you think it falls within this set of feature dimensions you've defined then maybe you'll get more human faces out the other end. I've obviously not tried this, and that's becuase there are obvious issues with labeling large datasets to this degree of complexity.

___

## Generative Adverserial Networks

So one way to solve the above problem is to ask the model to extract enough features from the dataset that make a face a face and then combine those somehow to generate a face. Here you don't stipulate the set of features or require they be exhastive, you just ask that it collect enough and combine them so as to be accurate. This means that you may end up ignoring parts of the data set and just focus on faces that have short hair for instance. By doing this we remove the constraint the naive approach suffers from. Namely that it minimzies loss between output and the whole dataset without any felxibility. Instead we just ask the model create something that passibly belongs to the dataset.

I'm going to focus on these types of model, known as Generative Adverserial Networks (GANs) but there are other methods such as autoencoders. Autoencoders take a different approach in that they use one network to compress the data into a low dimensional space and another to recreate it. In doing so you extract the smallest set of features the recreation network needs in order to reproduce the dataset. In this way you require the entire latent space to map to the entire dataset. This is similar in a sense to the niave approach except that instead of the two labels you use the higher dimensional continuum to give the model room to parameterise different features.

With a GAN you create two networks, one that generates images and one that tries to distinguish generated from real images. You then set them in compitition. So you ask the generator network to create a batch of fake images and then make there labels 0 as in False or Fake. You also sample a batch of real images with labels 1 as in True or real. You train the descriminater against this labelled data and in turn you train the generator by defining it's loss to be dependent on the number of generated images the descriminator correctly detected. So the generators trying to get as many fake images past the descriminator as possible.

For the generator we give as input a random point in some latent space. The idea here is that your telling it to map these points to faces and by giving it room to manuover it can generate as much varaition in faces as it can pack into the size of latent space you give it. The eventual mapping is going to be randomly assinged, so one area may end up encoding faces with glasses another without and another beards or smiles etc. We don't require that the generator be trying to reproduce the dataset in it's entirity instead we just want at least one instance of a face and this means that the generator may just decide to use the entire space to create a single instance of a face. This is known as mode collapse.

Here are the best results I got with this approach:

![simple-gan-results](/assets/generating-faces/simple-gan.png)

Yeah so not great! I tried this approach for a while before I got quite annoyed by the intermitency of what it seemed to be producing. Not really knowing what to expect from this learning process I gave up on this model and tried something I read about [here](https://arxiv.org/abs/1903.06048).

___

## Multiple Scale Gradient GANs

This approach maps the different resolutions of the data between the relevent layers of each of the generator and descriminator networks. This means the generator produces images of multiple different resolutions and the descriminator looks at each of these different resolution images and tries to figure out how real or fake they each are. I wouldn't say this is super clear to me what it's doing except that intuitivly it's preferentating learning lower resolution features before higher resolution ones. By building up a higherarchy of learnt features like this you aid stability in learning.

Anyway using this apporach we get the following significant improvement.

![msg-gan-face](/assets/generating-faces/msg-gan-face.png)

So this is a significant improvement on the inital attempt. It's still not perfect but it's enought to convince me that this really works! Training took a long time and I likely didn't run it long enough to get the best results I could. I think think long training times for these types of problems are expected but also my laptop is slooooooooowwwww.

___

## Analysis of Learning

So I think what ends up going on in the learning process must be that the descriminator network very  quicky picks up low resolution features on which to focus to detect face like qualities. This very quickly means that it can detect and differentate the random noise initally ouputted by the generator and the real faces in the dataset. In compition with this the generator has to work to match the set of features the descriminator is looking for. Once it's done so the descriminator now has to find a new feature to use in order to distinguish between the generated and real data. I think this must continue in a progression in which the each pair learns to detect and create seperate features. From watching the sequence of images immitted during training it would definetly seem like this porcess happens in order of feature resolution. So for intance the first thing the descriminator learns is that there is a grey blob in the middle of the screen, and then it starts to see darker blobs where the eyes and mouth would be, and so on until its learning to paint in the whites of the eyes. Becuase of this you'd expect the generator and descriminator loss functions to ossilate in compitition with each other. So when a new feature is discovered by the descriminator it should outperform the generator and when the generator learns to match the feature set the descriminator has derived it should push the descriminators loss up. This seems to be what happens:

![gans-losses](/assets/generating-faces/gans-losses.png)

The above illustraites the major frustration with these networks in that there is no strong measure of how much has been learnt becuase each of the above loss functions exists relative to the other. Hence the only thing you can really do is display a picture of a face at each stage of learning and decide whether or not it is more or less facey than previouse outputs. This is compounded by the fact that the network is learning over the input space so some areas by vitue of being less visited will be less trained and so a single generated image doesn't caputure everything the network has leant.

It also seems that becuase the generator is learning to fool the decriminator what it learns is very dependent on what the descriminator is choosing to focus on. It's not clear to me that the descriminator doesn't unlearn features if they no longer provide good indication of real or fake. For instance if the descriminator learns to detect noses as indicative of real images and in turn the generator learns to perfectly create noses to fool the derscriminator then when the descriminator moves on to some other feature to focus on I don't think there's any reason to assume it preseves what it's learnt about noses. It may sacrifice the nose knowledge in persut of the next feature. Clearly it must capture everything on average otherwise this method wouldn't work but by surveying the images the generator procduces over the course of training it seems like they sort of ossiclate in and out of levels of accuracy in different features and sometimes it's as if the genrerator and desciminator are sort of wandering over the dataset rather than learning to create a specific instance of something that might be in the dataset. This all makes it hard to know if its learning.

So yeah that basically concludes my eperience thus far using GANs. Here are some of the horrors summoned in the process of getting the above:

![screaming-ghost](/assets/generating-faces/screaming-ghost.png)
<img src="/assets/generating-faces/erm-msg-gan.png" alt="demon" width="100"/>
![angry-dude](/assets/generating-faces/angry-dude-msg-gan.png)
![demon](/assets/generating-faces/demon-msg--gan.png)
<img src="/assets/generating-faces/ghostly-chap.png" alt="demon" width="100"/>
<img src="/assets/generating-faces/black-eye.png" alt="demon" width="100"/>


Next I want to see how the descriminator has learnt features differently to how typical catigorization model do. You can get an idea of the features a network has learnt by giving it an image and asking it to change the image to maximise activation of certain layers. I tired this with model that I trained to discriminate happy and not-happy faces as a very naive attempt at generative modelling and was naturally disapointed. I Figured the set of features that these networks might be learning doesn't have to be the set of features you'd expect. Certainly it's going to ignore large amounts of the data if it's perhiperal to the task at hand. So it's not going to learn anything about a persons hair becuase that's common across photos of happy and not-happy people. Anyway it'll be interesting to do the same thing with the descriminator as it should have extracted a diverse range of features.

Now that I know what to expect from this process I also want to see how the above crosses over to the doodle example I was initially exploring. I Don't have super high expectations simply becuase the data set in that case is pretty small but we'll see what happens.
