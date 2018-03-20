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

I use the Nvidia's Model. I normalize and crop the image before put it into the network, works well for the first track. My first attempt to use generator result in a much slower traing time and worse training result. Because the input image is resized and my RAM is big enough to fit those training data, I just abandoned the generator in the first submition. But it's intresting to investigate that problem. I found that the 'steps_per_epoch' parameter was given a wrong number. It is given the number of the training samples. It should be devide by batch size. So when I set the batch size to 6, it actually run 6 epochs for one 'epoch', and is 6 times slower for that 'epoch'. After 3 'epochs', I actually run 18 epochs, which lead to overfitting and worse performance. After I devide the number by the batch size, everything works nice.
My final model consisted of the following layers:

| Layer                   | Description                                     |
| :---------------------: | :---------------------------------------------: |
| Input                   | 80x160x3 RGB image                              |
| Normalization           |                                                 |
| Cropping                |                                                 |
| Convolution 5x5x24      | 2x2 stride                                      |
| RELU                    |                                                 |
| BATCH NORM              |                                                 |
| Convolution 5x5x36      | 2x2 stride                                      |
| RELU                    |                                                 |
| BATCH NORM              |                                                 |
| Convolution 5x5x48      | 2x2 stride                                      |
| RELU                    |                                                 |
| BATCH NORM              |                                                 |
| Convolution 3x3x64      | 1x1 stride                                      |
| RELU                    |                                                 |
| BATCH NORM              |                                                 |
| Convolution 3x3x64      | 1x1 stride                                      |
| RELU                    |                                                 |
| BATCH NORM              |                                                 |
| Fully connected 100     |                                                 |
| BATCH NORM              |                                                 |
| DROPOUT                 |                                                 |
| Fully connected 50      |                                                 |
| BATCH NORM              |                                                 |
| DROPOUT                 |                                                 |
| Fully connected 10      |                                                 |
| BATCH NORM              |                                                 |
| DROPOUT                 |                                                 |
| Fully connected 1       |                                                 |


#### 2. Attempts to reduce overfitting in the model

The the portion of train and validation set is 0.8:0.2. My plan is start with the Nvidia's Model. And then apply some regulrazation method like dropout,and batch normalization to improve the model. But the model works too well,so that after 3 epochs training, the car is already able to drive pretty good on the first track. 
I add batch normalization after each activation. And dropout is added after each dense layer. I read the paper of batch normalization, they said batch normalization can provide some regulrazation effect. So I lower the dropout rate to 20%. It's interesting to see the difference with and without dropout. Below is the table of the training and validating result.
The result shows dropout version is significantly better for low variance between train and val set. The second epoch provide the best averange loss in validation set. So I use the weight recorded after second epoch.

| epochs      | BN only                     | BN+dropout                  |
| :---------: | :-------------------------: | :-------------------------: |
| 1 epoch     | 0.018/train , 0.0187/val    | 0.1036/train, 0.0166/val    |
| 2 epoch     | 0.013/train , 0.0204/val    | 0.017/train , 0.0156/val    |
| 3 epoch     | 0.0074/train, 0.0206/val    | 0.013/train , 0.0165/val    |


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 126).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on center of the road. I recorded two datasets,one used keyboard another used mouse. The data created by mouse offers a smoother driving manner in the result model.
I first recorded two laps, driving forward and backword on the track. After training on those data, the model fails to turn on the connors without the line ( see the example below ). So I recorded more data on that connor and trained the model again. Then the car drove nicely on those connors and able to drive 3 laps autonomously.

![alt text][image1]

### Simulation

The car is able to drive itself for at least 3 laps on the first track. The video.mp4 is a sample lap I recorded.
