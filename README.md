# Eye Tracker & gaz prediction

I) Preprocessing:

The preprocessing part was partly inspired by this tutorial:

https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html


2)Model and Results:

Implemented and improved the iTracker model proposed in the paper [Eye Tracking for Everyone](https://arxiv.org/abs/1606.05814).

In this modified model, the only difference between the modified model and the iTracker model is
that we concatenate the face layer FC-F1 and face mask layer FC-FG1 first, after applying a fully connected layer FC-F2,
we then concatenate the eye layer FC-E1 and FC-F2 layer.
This modified architecture is superior to the iTracker architecture.
Improvements in speed because the modified model converged faster 
28 epochs vs. 40+ epochs. In this case a smaller number of epochs is enaugh to converge to acceptable level of accuracy.
Improvements in accuracy and achieved better validation error 
2.19 cm vs. 2.51 cm).
In fact, makes the model faster because the model only takes the relevant information from the eye.

The iTracker model was implemented in itracker.py and the modified one was implemented in itracker_adv.py.
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

2) Python 3 Installation & Setup Guide:
Please follow these instructions for installing Python packaging:
https://packaging.python.org/tutorials/installing-packages/

3) For TensorFlow installation:
https://www.tensorflow.org/install

4) For opencv packagin installation:
https://pypi.org/project/opencv-python/




