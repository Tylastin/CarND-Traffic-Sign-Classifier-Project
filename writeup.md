# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization_training.png "Training Visualization"
[image2]: ./examples/visualization_validation.png "Validation Visualization"
[image3]: ./examples/visualization_test.png "Test Visualization"
[image4]: ./examples/preprocessed_image.png "Preprocessed Image"
[image5]: ./examples/normalized_grayscale.png "Normalizing and Grayscaling"

[image6]: .examples/german1.png "Double Curve Sign"
[image7]: ./examples/german2.png "Keep Right Sign"
[image8]: ./examples/german3.png "Speed Limit (60km/h) Sign"
[image9]: ./examples/german4.png "Speed Limit (30km/h) Sign"
[image10]: ./examples/german5.png "Yield Sign"

[image11]: .../data/examples/softmax_output.jpg "Softmax Probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Tylastin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a set of 3 bar charts showing the frequency of each sign type for the training set, validation set, and test set.

![alt text][image1]
![alt text][image2]
![alt text][image3]
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color in the images seemed unimportant for the classification task, and reducing the dimensions would make training more tractable. As a second step, I normalized the image data because it ensures that all pixel values are in a similar range, which makes the network converge faster.

Here is an example of a traffic sign image before processing.

![alt text][image4]


Here is an example of a normalized grayscale image

![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	2x2     	| 2x2 stride, valid padding,  outputs 15x15x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x64	     				
| RELU					|												|
| Max pooling	2x2     	| 1x1 stride, valid padding,  outputs 12x12x64 	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x128	     	|			
| RELU					|												|
| Max pooling	2x2     	| 2x2 stride, valid padding,  outputs 5x5x128 	|
| Flatten          |  outputs 3200 |
| Fully connected		|  outputs 120 |
| RELU					|										|
| Fully connected		|  outputs 84   		|
| RELU					|												|
| Dropout					|		50%	keep_rate						|
| Fully connected		|  outputs 43 ( logits )		|
| Softmax				|      outputs 43 ( probabilities )	|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer with the following parameters:
batch size: 100,
epochs: 50
learning rate: 0.0009
mu = 0 (mean value for tf.truncated_normal)
sigma = 0.1 (standard deviation value for tf.truncated_normal)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 94.9%
* test set accuracy of 93.4%


* What was the first architecture that was tried and why was it chosen?

The first architecture that was tried was the LeNet-5 architecture. It was chosen because it has proven effective for image classification.

* What were some problems with the initial architecture?

The architecture performed reasonably well but the validation accuracy was well below the 93% requirement.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

A total of three 3x3 convolution (conv2d, relu, max_pooling) layers ( with more features than the LeNet architecture) were used allow better fitting of the data. A dropout was added to one of the fully connected layers to prevent overfitting.

* Which parameters were tuned? How were they adjusted and why?
The batch size, number of epochs, learning rate were tuned to achieve higher validation accuracy. They were adjusted iteratively, and the most optimally performing values were kept.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layers were used because they are effective for image classification due to translational invariance. The dropout layer was added because it prevents overfitting by forcing the model not to rely too heavily on a few features.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]

The chosen images are difficult to classify because the images are of different sizes and the signs in the are off-center and at different angles. Also, some of the traffic signs such as double curve were underrepresented in the training set.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set


Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Double Curve      		| No Passing   									|
| Keep Right     			| Turn Right Ahead								|
| 60 km/h						| 60 km/h											|
| 30 km/h	      		| 60 km/h						 				|
| Yield			| Turn Right Ahead      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares unfavorably to the accuracy on the test set of 93.4%. Augmented the dataset to add more images from the underrepresented classes could improve performance. Also augmenting the training set with random transformations such as translation and scaling could further improve performance.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the final cell of the Ipython notebook.

The image below shows the softmax probabilities for each german traffic sign. The model was only able to correctly classify the 60 km/h sign and it did so with 71% certainty.
Here is an image of the output
![alt text][image11]
