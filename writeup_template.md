# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./results/class_distribution.png "Visualization"
[image2]: ./results/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./german_traffic_signs/30kph.png "Traffic Sign 1"
[image5]: ./german_traffic_signs/stop.png "Traffic Sign 2"
[image6]: ./german_traffic_signs/stop1.png "Traffic Sign 3"
[image7]: ./german_traffic_signs/straight.png "Traffic Sign 4"
[image8]: ./german_traffic_signs/vorfahrt.png "Traffic Sign 5"
[image9]: ./german_traffic_signs/pedestrians.png "Traffic Sign 3"
[image10]: ./german_traffic_signs/vorfahrt1.png "Traffic Sign 4"
[image11]: ./german_traffic_signs/right.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing I am running two different steps. First I grayscale the images. This should speed up the training and prediction since the color usually doesnt play a big role for neural networks. In the second step the iamges are normalized so that the data has mean zero and equal variance. In addition I used the histogram normalization technique to correct pixel intensity values.

The resulting images are displayed below.

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model has the following architecture:

| Layer  | Description  |
|---|---|
|Input|	32x32x1 Grayscale image|
|Convolution 5x5|	3x3 filters, 1x1 stride, valid padding, outputs 28x28x6|
|RELU|	|
|Max Pooling |	2x2 stride, outputs 14x14x6|
|Convolution 5x5|	3x3 filters, 1x1 stride, valid padding, outputs 10x10x16|
|RELU|	|
|Max Pooling |	2x2 stride, outputs 5x5x16|
|Flatten|	outputs 400|
|Fully connected| outputs 120|
|RELU|	|
|Dropout | keep prop 0.5|
|Fully connected| outputs 84|
|RELU|	|
|Dropout | keep prop 0.5|
|Fully connected| outputs 43|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model a Adam Optimizer was used with a learning rate of 0.0015. The learning rate is decaying every 10000 steps by 0.9. I trained for 50 epochs with a batch size of 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.953
* test set accuracy of 0.929

As proposed I am using the LeNet architecture from the lecture. This already provided a pretty good accuracy. Since a decent overfitting could be recognized I added to dropout layers before the fully connected layers. This help to boost the validation set accuracy to 95.3%. 

The only hyperparameter which was tuned, was the learning rate. The leraning rate from the lessions was 0.001, I start with a slightly higher learning rate of 0.0015 and then add a exponential decay of 0.9 every 10000 steps.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] 

The first image might be difficult to classify because the number inside the sign is written in a relatively small size and is enclosed in a panel.

The second, third and fourth image is pretty clear readable and looks to have decent enough image information to be properly detected.

The fifth and sixth image are pretty occluded and even for humand pretty hard to read, it is to some extend expected for the neural net that this is not detected properly.

The last image is a little skewed so it could be that the NN cannot detect this sign properly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Dangerous curve to the right   				| 
| Stop       			| Stop 										    |
| No entry				| No entry										|
| Ahead only	      	| Ahead only					 				|
| Priority road			| Priority road      							|
| Pedestrians           | General caution                               |
| Priority road			| Priority road      							|
| Turn right ahead		| Ahead only      						    	|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75.0%. This compares slightly less accurate then on the test of 91.6%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a Keep right sign (probability of 1.00), but the image acutally is a 30 kph sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right    								| 
| .00     				| Roundabout mandatory 							|
| .00					| Children crossing								|
| .00	      			| Dangerous curve to the right					|
| .00				    | Go straight or right      					|

For the second image the model again is sure that it is a stop sign, in this case it is correct

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop    								| 
| .00     				| Speed limit (60km/h) 							|
| .00					| Speed limit (80km/h)								|
| .00	      			| Speed limit (30km/h)					|
| .00				    | General caution      					|

Same holds true for the third image, again this is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry    								| 
| .00     				| Stop 							|
| .00					| No passing								|
| .00	      			| Turn right ahead					|
| .00				    | Dangerous curve to the right      					|

Also for the ahead only image the model correctly detects the sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only    								| 
| .00     				| Turn right ahead 							|
| .00					| End of no passing								|
| .00	      			| Priority road					|
| .00				    | Go straight or left      					|

Also the priority road sign is detected correctly twice for different images.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority road    								| 
| .00     				| Yield 							|
| .00					| Ahead only								|
| .00	      			| Roundabout mandatory					|
| .00				    | End of no passing      					|

I even tested more images, check the notebook for other images. The pedestrians crossing sign was not detected correctly, with only 0.479%, but insted was detected as Traffic signals with 60.932%. The last sign, Turn right ahead, was this time detected correctly, but in a previous run was detected as a different sign. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


