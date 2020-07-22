---
layout: post
title:  "Machine Learning, Recovering Obscured Handwritten Numbers"
date:   2020-07-15 00:00:00 +0100
categories: machine-learning
---

___

So this was mostly just a test to see how well we can train a model to take obscured handwriting and output a cleaned version of the image. I was just starting out at this stage coming to understand the kinds of models that machine learning allows you to build. I wanted to see if this would be as easy as I thought it would be.  

I'm using the MNIST handwritten digits data set. For each image I'm going to build a pipeline that spits out as input for the model an image that's been partially obscured by blobs of different sizes. The label that we want the model to be matching against is then just the unblemished article.  

This is some typical output on a unseen test data set. The first column is the handwritten digits with randomly positioned spheres on top. The second is the output from the neural network after training and the third is the unblemished letters.

<img src="/assets/unobscuring-handwritten-letters/blobbed-numbers.png" alt="demon" height="220" width="40"/>
<img src="/assets/unobscuring-handwritten-letters/unblobbed-numbers.png" alt="demon" height="220" width="40"/>
<img src="/assets/unobscuring-handwritten-letters/clean-numbers.png" alt="demon" height="220" width="40"/>

So this worked pretty well and was super fast. This whole experience was pretty encouraging and also I'm afraid completely unrepresentative of everything that followed.
