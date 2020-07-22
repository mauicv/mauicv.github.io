---
layout: post
title:  "Machine Learning, Computer Generated Faces"
date:   2020-05-25 18:37:37 +0100
categories: machine-learning
---

___

## Motivation

I started my Computer Generated pictures adventure by looking at trying to get the computer to generate images of doodles I'd drawn. I have this giant doodle I'd been noodling away on for a while and by taking a set of pictures of it and then subsampling bits of it I hoped that this would be enough to generate new doodles in the same style. I got a little lost. Remember I'm new to all this stuff and the main thing I'm coming up against is not really knowing what  to expect from the process of computer learning. Like how long before you give up on a model learning something, how a model can fail, what it looks like for a model to fail and so on. I'm intending on coming back to the doodle example but in order to better understanding what was going on I decided to opt for a dataset which better benchmarks, namely faces. The sole aim here is to end up with a computer generated picture of a face that is at least vaguely believably human.

___

## Naive Approach

So the first question I had was why can you not just take train something like a reversed categorization model. So instead of taking images are returning a category you take a category and return an image. The principle should be the same you just reverse the images and instead of computing the categorical cross entropy between the predicted category and the real one you use a loss function that computes the difference between two images. I tried this out and I got this sightly disturbing pair.

![happy-and-sad](/assets/generating-faces/happy-and-sad.png)

The main thing to note here is that it's obviously generating an image that's in some sense an average of all smiley faces or non-smiley faces rather than a specific instance of a smiley face. This makes sense because there is only one input either 0 or 1 and it's expected to map each of these values into an image that best represents a wide range of pictures of faces. If it where to create a face that is very face like then it may be close to some faces within that set of faces but it will necessarily also be far away from others. So instead of a instance of an identifiably human face you instead get a eerie blurred mask that represents lots of faces all at once.

So the I kind of expected this would happen. The issue is that the inputs don't give any room for manoeuvrer. Your asking these two values to describe the entirety of the dataset so of course the output will be a blended version of all the images. The natural solution to this is to make the space that represents the faces larger. There are some obvious difficulties to navigate here. Suppose you had a better labelling system. So instead of 0 or 1 lets make it a continuum. In the above example 0 was not-happy and 1 was happy. So with a continuum we could represent a sliding scale of happiness where some faces are more happy than other faces. Then we can add  other dimensions. So instead of just a happiness dimension we can have a hair dimension and a nose size dimension and so on... If your willing to spend the time going through the data set and labelling each image by where you think it falls within this set of feature dimensions you've defined then maybe you'll get more human faces out the other end. I've obviously not tried this, and that's because there are obvious issues with labelling large datasets to this degree of complexity.

___

## Generative Adversarial Networks

So one way to solve the above problem is to ask the model to extract enough features from the dataset that make a face a face and then combine those somehow to generate a face. You don't stipulate the set of features or require they be exhaustive, you just ask that it collect enough and combine them so as to be accurate. This means that you may end up ignoring parts of the data set and just focus on faces that have short hair for instance. By doing this we remove the constraint the naive approach suffers from. Namely that it minimizes loss between output and the whole dataset. Instead we just ask the model create something that passably belongs to the dataset.

I'm going to focus on these types of model, known as Generative Adversarial Networks (GANs). With a GAN you create two networks, one that generates images and one that tries to distinguish generated from real images. You then set them in competition. So you ask the generator network to create a batch of fake images and then make there labels 0 as in False or Fake. You also sample a batch of real images with labels 1 as in True or real. You train the discriminator against this labelled data and in turn you train the generator by defining it's loss to be dependent on the number of generated images the discriminator correctly detected. So the generators trying to get as many fake images past the discriminator as possible.

For the generator we give as input a random point in some latent space. The idea here is that your telling it to map these points to faces and by giving it room to manoeuvrer it can generate as much variation in faces as it can pack into the size of latent space you give it. The eventual mapping is going to be randomly assigned, so one area may end up encoding faces with glasses another without and another beards or smiles and so on... We don't require that the generator be trying to reproduce the dataset in it's entirety instead we just want at least one instance of a face and this means that the generator may just decide to use the entire space to create a single instance of a face. This is known as mode collapse and it's an issue generally with GANs but in my case if we get something human looking I'm happy.

Here are the best results I got with this approach:

![simple-gan-results](/assets/generating-faces/simple-gan.png)

Yeah so not great! I tried this approach for a while before I got quite annoyed by the intermittency of what it seemed to be producing. Not really knowing what to expect from this learning process I gave up on this model and tried something I read about [here](https://arxiv.org/abs/1903.06048).

___

## Multiple Scale Gradient GANs

This approach maps the different resolutions of the data between the relevant layers of each of the generator and discriminator networks. This means the generator produces images of multiple different resolutions and the discriminator looks at each of these different resolution images and tries to figure out how real or fake they each are. I wouldn't say this is super clear to me what it's doing except that intuitively it makes the network prefer learning lower resolution features before building higher resolution ones on top. By building up a hierarchy of learnt features like this you aid stability in learning.

Anyway using this approach I started getting stuff that was actually passible.

![bald-man-with-glasses](/assets/generating-faces/bald-man-with-glasses.png)
![happy-chappy](/assets/generating-faces/happy-chappy.png)
![camera-flash](/assets/generating-faces/camera-flash.png)
![big-head](/assets/generating-faces/big-head.png)



![msg-gan-faces](/assets/generating-faces/tiled-faces-msg-gan.png)
![msg-gan-faces](/assets/generating-faces/tiled-faces-msg-gan-2.png)

So this is a significant improvement on the initial attempt. I was pleased with these results but do note that they're not where near close to [what's possible](https://thispersondoesnotexist.com/)! Training took a long time which I think is typical but also my laptop is slooooooooowwwww.

___

## How it Learns

This section is hypothesis but I think what ends up going on in the learning process must be that the discriminator network picks up low resolution features on which to focus to detect face like qualities. This very quickly means that it can detect and differentiate the random noise initially outputted by the generator and the real faces in the dataset. In competition with this the generator has to work to match the set of features the discriminator is looking for. Once it's done so the discriminator now has to find a new feature to use in order to distinguish between the generated and real data. I think this must continue in a progression in which the each pair learns to detect and create separate features. From watching the sequence of images emitted during training it would definitely seem like this process happens in order of feature resolution. So for instance the first thing the discriminator learns is that there is a grey blob in the middle of the screen, and then it starts to see darker blobs where the eyes and mouth would be, and so on until its furnishing the finer details such as painting in the whites of the eyes. Because of this you'd expect the generator and discriminator loss functions to oscillate in competition with each other. So when a new feature is discovered by the discriminator it should outperform the generator and when the generator learns to match the feature set the discriminator has derived it should push the discriminators loss up. This seems to be what happens:

![gans-losses](/assets/generating-faces/gans-losses.png)

The above also illustrates the major frustration with these networks in that there is no strong measure of how much has been learnt because each of the above loss functions exists relative to the other. Hence the only thing you can really do is display a picture of a face at each stage of learning and decide whether or not it is more or less facey than previous outputs. This is compounded by the fact that the network is learning over the input space so some areas by virtue of being less visited will be less trained and so a single generated image doesn't capture everything the network has leant.

It also seems that because the generator is learning to fool the discriminator what it learns is very dependent on what the discriminator is choosing to focus on. It's not clear to me that the discriminator doesn't unlearn features if they no longer provide good indication of real or fake. For instance if the discriminator learns to detect noses as indicative of real images and in turn the generator learns to perfectly create noses to fool the discriminator then when the discriminator moves on to some other feature to focus on I don't think there's any reason to assume it preserves what it's learnt about noses. It may sacrifice the nose knowledge in pursuit of the next feature. Clearly it must capture everything on average otherwise this method wouldn't work but by surveying the images the generator produces over the course of training it seems like they sort of oscillates in and out of levels of accuracy in different features and sometimes it's as if the generator and discriminator are sort of wandering over the dataset rather than focusing on a particular member of the dataset. This all makes it hard to know if its learning.

So yeah that basically concludes my experience thus far using GANs. One final thing however that I was not entirely prepared for. As the generator gets better and better at producing pictures of faces it also spends more time wondering around in the uncanny valley. Here are some of the horrors summoned in the process of getting the above:

![screaming-ghost](/assets/generating-faces/screaming-ghost.png)
<img src="/assets/generating-faces/erm-msg-gan.png" alt="demon" width="100"/>
![angry-dude](/assets/generating-faces/angry-dude-msg-gan.png)
![demon](/assets/generating-faces/demon-msg--gan.png)
<img src="/assets/generating-faces/ghostly-chap.png" alt="demon" width="100"/>
<img src="/assets/generating-faces/black-eye.png" alt="demon" width="100"/>

![skeletal](/assets/generating-faces/skeletal.png)
![uncanny-valley](/assets/generating-faces/uncanny-valley.png)
![weird-eyes](/assets/generating-faces/weird-eyes-1.png)
![locals](/assets/generating-faces/locals.png)

___

##  Next steps

Next I want to see how the discriminator has learnt features differently to how typical categorization model do. You can get an idea of the features a network has learnt by giving it an image and asking it to change the image to maximise activation of certain layers. I tired this with model that I trained to discriminate happy and not-happy faces as a very naive attempt at generative modelling and was disappointed by the results. I Figured the set of features that these networks might be learning doesn't have to be the set of features you'd expect. Certainly it's going to ignore large amounts of the data if it's peripheral to the task at hand. So it's not going to learn anything about a persons hair because that's common across photos of happy and not-happy people. Anyway it'll be interesting to do the same thing with the discriminator trained above as it should have extracted a diverse range of features.

Now that I know what to expect from this process I also want to see how the above crosses over to the doodle example I was initially exploring. I Don't have super high expectations simply because the data set in that case is pretty small but we'll see what happens.
