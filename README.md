# Overview 

### Motivation
This project aims to put the power of eye tracking in everyone's palm, by building a software that works on commodity hardware such as mobile phones and tablets, without the need for additional sensors or devices. 

Gaze prediction is able to capture unbiased subconcious information, and is important to many domains, including:
 - Human–computer interaction
 - Medical diagnoses
 - Psychological studies
 - Computer vision
 - Augmented and Virtual reality
 - Robotics
 - Military / Aerospace
 - Smart and Wearable devices
 - Lie detection
 - Video games
 - Marketing research

# Eye Tracker & Gaze Prediction
=======
# Solution

I have utilized an eye tracking CNN models algorithm results from each model to bluid an eye tracking model which achieves a significant reduction in error over previous approaches while running in real time (10–15fps) on a modern mobile device.
input to the model: 
(1) the image of the face together with its location in the image (termed face grid), and (2) the image of the eyes. 
The output is the distance, in centimeters, from the camera.

<img width="960" alt="image" src="https://user-images.githubusercontent.com/41544179/54061096-21e3b280-41b4-11e9-96d9-463ac9ed25ad.png">

(1) the image of the face together with its location in the image (termed face grid), and (2) the image of the eyes. 
The output is the distance, in centimeters, from the camera.

# Data-driven approach:

In this project, I used a subset of the GazeCapture, containing data from over 1450 people consisting of almost 1:5M frames from a wide variety of backgrounds, recorded under variable lighting conditions and unconstrained head motion.
And here we have the preprocessing : first : Load image, 2) convert to grayscale, 3) find face, and find eye. 
The extraction of Face and eye extraction took:  0.032s (my code).

<img width="619" alt="image" src="https://user-images.githubusercontent.com/41544179/54061325-36747a80-41b5-11e9-909c-fb95b7fc2fd4.png">


1) Preprocessing:

The preprocessing is partly inspired by this tutorial:

https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html


2)Model and Results:

The project implements and improves the iTracker model proposed in the paper [Eye Tracking for Everyone](https://arxiv.org/abs/1606.05814).

In this modification, the face layer FC-F1 and face mask layer FC-FG1 are concatenated first, after applying a fully connected layer FC-F2. The eye layer FC-E1 and FC-F2 layer are then concatenated.

This improves the convergence speed from the iTracker model from 40+ epochs to 28 to converge to acceptible accuracy.
This model also improves validation error from 2.51 cm to 2.19 cm, compared to the iTracker model.
These improvements are the result of the model only taking relevant information from the eye, rather than the entire face.

The iTracker model is implemented in itracker.py.
The modified model is implemented in itracker_adv.py.
Note that a smaller dataset (i.e., a subset of the full dataset in the original paper) was used in experiments and no data augmentation was applied.
This smaller dataset contains 48,000 training samples and 5,000 validation samples.

# Get started
To train the model: run
`python itracker_adv.py --train -i input_data -sm saved_model`

To test the trained model: run
`python itracker_adv.py -i input_data -lm saved_model`

You can find a pretrained (on the smaller dataset) model under the pretrained_models/itracker_adv/ folder.

# FAQ
1) What are the datasets?

The original dataset comes from the [GazeCapture](http://gazecapture.csail.mit.edu/) project. The dataset involves over 1400 subjects and results in more than 2 million face images. Due to the limitation of computation power, a much [smaller dataset] with 48000 training samples and 5000 validation samples was used here. Each sample contains 5 items: face, left eye, right eye, face mask and labels.

2) Python Installation & Setup Guide:
Please follow these instructions for installing Python packaging:
https://packaging.python.org/tutorials/installing-packages/

3) For TensorFlow installation:
https://www.tensorflow.org/install

4) For opencv packaging installation:
https://pypi.org/project/opencv-python/




