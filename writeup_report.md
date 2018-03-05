# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/ex1.jpg "fail example"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the Nvidia's Model. I normalize and crop the image before put it into the network, works well for the first track.
My final model consisted of the following layers:

| Layer                   | Description                                     |
| :---------------------: | :---------------------------------------------: |
| Input                   | 80x160x3 RGB image                              |
| Normalization           |                                                 |
| Cropping                |                                                 |
| Convolution 5x5x24      | 2x2 stride                                      |
| RELU                    |                                                 |
| Convolution 5x5x36      | 2x2 stride                                      |
| RELU                    |                                                 |
| Convolution 5x5x48      | 2x2 stride                                      |
| RELU                    |                                                 |
| Convolution 3x3x64      | 1x1 stride                                      |
| RELU                    |                                                 |
| Convolution 3x3x64      | 1x1 stride                                      |
| RELU                    |                                                 |
| Fully connected 100     |                                                 |
| Fully connected 50      |                                                 |
| Fully connected 10      |                                                 |
| Fully connected 1       |                                                 |


#### 2. Attempts to reduce overfitting in the model

The the portion of train and validation set is 0.8:0.2. My plan is start with the Nvidia's Model. And then apply some regulrazation method like dropout,and batch normalization to improve the model. But the model works too well,so that after 3 epochs training, the car is already able to drive pretty good on the first track. I'm too lazy to add any of the regulrazation code after that.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 68).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on center of the road. I recorded two datasets,one used keyboard another used mouse. The data created by mouse offers a smoother driving manner in the result model.
I first recorded two laps, driving forward and backword on the track. After training on those data, the model fails to turn on the connors without the line ( see the example below ). So I recorded more data on that connor and trained the model again. Then the car drove nicely on those connors and able to drive 3 laps autonomously.
![alt text][image1]

### Simulation

The car is able to drive itself for at least 3 laps on the first track. The run1.mp4 is a sample lap I recorded.
